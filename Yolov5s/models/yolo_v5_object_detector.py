import numpy as np
from deep_utils.utils.box_utils.boxes import Box
import torch
# import os
from models.experimental import attempt_load
from utils.general import xywh2xyxy
from utils.datasets import letterbox
import cv2
import time
import torchvision
import torch.nn as nn
from utils.metrics import box_iou


class YOLOV5TorchObjectDetector(nn.Module):
    def __init__(self,
                 model_weight,
                 sel_nms,
                 sel_prob,
                 device,
                 img_size,
                 names=None,
                 mode='eval',
                 confidence=0.4,
                 iou_thresh=0.45,
                 agnostic_nms=False):
        super(YOLOV5TorchObjectDetector, self).__init__()
        self.device = device
        self.model = None
        self.sel_nms = sel_nms
        self.sel_prob = sel_prob
        self.img_size = img_size
        self.mode = mode
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.agnostic = agnostic_nms
        self.model = attempt_load(model_weight, map_location=device)
        print("[INFO] Model is loaded")
        self.model.requires_grad_(True)
        self.model.to(device)
        if self.mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        # fetch the names
        if names is None:
            print('[INFO] fetching names from bdd100k-self file')
            self.names = ['person', 'rider', 'car', 'bus', 'truck']
        else:
            self.names = names

        # preventing cold start
        img = torch.zeros((1, 3, *self.img_size), device=device)
        self.model(img)

    @staticmethod
    def non_max_suppression(prediction, prediction_, logits, classHead, sel_nms, sel_prob, conf_thres=0.6, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False, labels=(), max_det=300):
        """Runs Non-Maximum Suppression (NMS) on inference and logits results

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls] and pruned input logits (n, number-classes)
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        logits_output = [torch.zeros((0, 80), device=logits.device)] * logits.shape[0]
        obj_conf_output = [torch.zeros((0, 1), device=prediction_.device)] * prediction_.shape[0]
        classHead_output = [torch.zeros((0, 1), device=classHead.device)] * classHead.shape[0]
        for xi, (x, x_, log_, classHead_) in enumerate(zip(prediction, prediction_, logits, classHead)):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            x_ = x_[xc[xi]]
            #x_ = x_[:, 4:5]
            x_ = x_[:, 4:5]
            log_ = log_[xc[xi]]

            log_ = torch.sigmoid(log_)
            x_ = torch.sigmoid(x_)

            if sel_prob == 'objclass':
                log_ = log_ * x_  # conf = obj_conf * cls_conf

            classHead_ = classHead_[xc[xi]]

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # log_ *= x[:, 4:5]
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                # log_ = x[:, 5:]
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
                x_ = x_[conf.view(-1) > conf_thres]
                log_ = log_[conf.view(-1) > conf_thres]
                classHead_ = classHead_[conf.view(-1) > conf_thres]
            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            if sel_nms == 'NMS':
                output[xi] = x[i]
                obj_conf_output[xi] = x_[i]
                logits_output[xi] = log_[i]
                classHead_output[xi] = classHead_[i]
            else:
                output[xi] = x
                obj_conf_output[xi] = x_
                logits_output[xi] = log_
                classHead_output[xi] = classHead_

            assert log_[i].shape[0] == x[i].shape[0]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output, logits_output, obj_conf_output, classHead_output

    @staticmethod
    def yolo_resize(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):

        return letterbox(img, new_shape=new_shape, color=color, auto=auto, scaleFill=scaleFill, scaleup=scaleup)

    def forward(self, img):
        prediction, full_logits = self.model(img, augment=False)
        full_logits_cat = torch.cat((torch.flatten(full_logits[0], 1,3), torch.flatten(full_logits[1], 1,3), torch.flatten(full_logits[2], 1,3)), 1)
        classHead = torch.cat((torch.flatten(torch.zeros(full_logits[0].shape).mean(dim=4)+2, 1,3), torch.flatten(torch.zeros(full_logits[1].shape).mean(dim=4)+1, 1,3), torch.flatten(torch.zeros(full_logits[2].shape).mean(dim=4)+0, 1,3)), 1)
        pred_conf_logits = full_logits_cat[:,:,0:5]
        logits = full_logits_cat[:,:,5:]

        prediction, logits, obj_confs, classHead_output = self.non_max_suppression(prediction, pred_conf_logits, logits, classHead, self.sel_nms, self.sel_prob, self.confidence, self.iou_thresh,
                                                      classes=None,
                                                      agnostic=self.agnostic)
        self.boxes, self.class_names, self.classes, self.confidences = [[[] for _ in range(img.shape[0])] for _ in
                                                                        range(4)]
        for i, det in enumerate(prediction):  # detections per image
            if len(det):
                for *xyxy, conf, cls in det:
                    bbox = Box.box2box(xyxy,
                                       in_source=Box.BoxSource.Torch,
                                       to_source=Box.BoxSource.Numpy,
                                       return_int=True)
                    self.boxes[i].append(bbox)
                    self.confidences[i].append(round(conf.item(), 2))
                    cls = int(cls.item())
                    self.classes[i].append(cls)
                    if self.names is not None:
                        self.class_names[i].append(self.names[cls])
                    else:
                        self.class_names[i].append(cls)
        # tmp = full_logits[0][0, 0, 0, 0, 0]
        return [self.boxes, self.classes, self.class_names, self.confidences], logits, obj_confs, classHead_output

    def preprocessing(self, img):
        if len(img.shape) != 4:
            img = np.expand_dims(img, axis=0)
        im0 = img.astype(np.uint8)
        img = np.array([self.yolo_resize(im, new_shape=self.img_size)[0] for im in im0])
        img = img.transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img / 255.0
        return img


if __name__ == '__main__':
    model_path = 'runs/train/cart-detection/weights/best.pt'
    img_path = './16_4322071600_101_0_4160379257.jpg'
    model = YOLOV5TorchObjectDetector(model_path, 'cpu', img_size=(640, 640)).to('cpu')
    img = np.expand_dims(cv2.imread(img_path)[..., ::-1], axis=0)
    img = model.preprocessing(img)
    a = model(img)
    print(model._modules)
