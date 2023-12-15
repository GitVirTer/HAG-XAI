import time
import torch
import torch.nn.functional as F
import gc
import util_my_yolov5 as ut
import numpy as np

def find_yolo_layer(model, layer_name):
    """Find yolov5 layer to calculate GradCAM and GradCAM++

    Args:
        model: yolov5 model.
        layer_name (str): the name of layer with its hierarchical information.

    Return:
        target_layer: found layer
    """
    hierarchy = layer_name.split('_')
    target_layer = model.model._modules[hierarchy[0]]

    for h in hierarchy[1:]:
        target_layer = target_layer._modules[h]
    return target_layer


class YOLOV5XAI:

    def __init__(self, model, layer_names, sel_prob_str, sel_norm_str, sel_classes, sel_XAImethod, img_size=(640, 640)):
        self.model = model
        self.gradients = dict()
        self.activations = dict()
        self.sel_prob_str = sel_prob_str
        self.sel_norm_str = sel_norm_str
        self.sel_classes = sel_classes
        self.sel_XAImethod = sel_XAImethod

        def backward_hook_0(module, grad_input, grad_output):
            self.gradients[0] = grad_output[0]
            return None
        def forward_hook_0(module, input, output):
            self.activations[0] = output
            return None

        def backward_hook_1(module, grad_input, grad_output):
            self.gradients[1] = grad_output[0]
            return None
        def forward_hook_1(module, input, output):
            self.activations[1] = output
            return None

        def backward_hook_2(module, grad_input, grad_output):
            self.gradients[2] = grad_output[0]
            return None
        def forward_hook_2(module, input, output):
            self.activations[2] = output
            return None

        target_layer = find_yolo_layer(self.model, layer_names[0])
        target_layer.register_forward_hook(forward_hook_0)
        target_layer.register_backward_hook(backward_hook_0)
        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))
        print('[INFO] saliency_map size :', self.activations[0].shape[2:])

        target_layer = find_yolo_layer(self.model, layer_names[1])
        target_layer.register_forward_hook(forward_hook_1)
        target_layer.register_backward_hook(backward_hook_1)
        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))
        print('[INFO] saliency_map size :', self.activations[1].shape[2:])

        target_layer = find_yolo_layer(self.model, layer_names[2])
        target_layer.register_forward_hook(forward_hook_2)
        target_layer.register_backward_hook(backward_hook_2)
        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))
        print('[INFO] saliency_map size :', self.activations[2].shape[2:])



    def forward(self, input_img, img, pred_label, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        class_prob_list = []
        head_num_list = []
        nObj = 0
        b, c, h, w = input_img.size()
        tic = time.time()
        preds, logits, preds_logits, classHead_output = self.model(input_img)
        classHead_output = classHead_output[0].detach().numpy()
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        # class_probs = torch.sigmoid(logits[0])
        raw_data_rec = []
        class_probs = (logits[0])
        pred_list = []
        pred_list.append([])
        pred_list.append([])
        pred_list.append([])
        pred_list.append([])
        with torch.autograd.set_detect_anomaly(True):
            for pred_logit, logit, bbox, cls, cls_name, obj_prob, class_prob, classHead in zip(preds_logits[0], logits[0], preds[0][0], preds[1][0], preds[2][0], preds[3][0], class_probs, classHead_output):
                img_processed = input_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2,
                                                                                               0).detach().cpu().numpy()
                shape_raw = [np.size(img_processed, 1), np.size(img_processed, 0)]  # w, h
                shape_new = [np.size(img, 1), np.size(img, 0)]  # w, h
                boxes_rescale_xyxy, boxes_rescale_xywh, _ = ut.rescale_box_list([[bbox]], shape_raw, shape_new)

                if (boxes_rescale_xywh == pred_label).all():
                    if self.sel_prob_str == 'obj':
                        score = pred_logit
                    else:
                        score = logit[cls]
                    class_prob_score = class_prob[cls]
                    class_prob_score = torch.unsqueeze(class_prob_score, 0)
                    class_prob_list.append(class_prob_score)

                    pred_list[0].append([bbox])
                    pred_list[1].append([cls])
                    pred_list[2].append([cls_name])
                    pred_list[3].append([obj_prob])

                    self.model.zero_grad()
                    tic = time.time()
                    score.backward(retain_graph=True)
                    print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')

                    head_num_list.append([classHead])
                    gradients = self.gradients[classHead]
                    activations = self.activations[classHead]

                    if self.sel_XAImethod == 'gradcam':
                        saliency_map = ut.gradcam_operation(activations, gradients)
                    elif self.sel_XAImethod == 'gradcampp':
                        saliency_map = ut.gradcampp_operation(activations, gradients)
                    elif self.sel_XAImethod == 'fullgradcam':
                        saliency_map = ut.fullgradcam_operation(activations, gradients)
                    elif self.sel_XAImethod == 'fullgradcamraw':
                        saliency_map = ut.fullgradcamraw_operation(activations, gradients)
                    elif self.sel_XAImethod == 'fullgradcamneg':
                        saliency_map = ut.fullgradcamneg_operation(activations, gradients)
                    elif self.sel_XAImethod == 'saveRawGradAct':
                        saliency_map, raw_data = ut.saveRawGradAct_operation(activations, gradients)
                        raw_data_rec.append(raw_data)

                    saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                    if self.sel_norm_str == 'norm':
                        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

                    nObj = nObj + 1
                    if nObj == 1:
                        saliency_map_sum = saliency_map
                    else:
                        saliency_map_sum = saliency_map_sum + saliency_map

                # pred_logit = pred_logit.detach()
                # logit = logit.detach()
                # class_prob = class_prob.detach()

        if nObj == 0:
            saliency_map_sum = torch.zeros([1, 1, h, w])
        else:
            saliency_map_sum = saliency_map_sum / nObj

        #saliency_map_sum = F.relu(saliency_map_sum)
        saliency_map = saliency_map_sum
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        saliency_map = saliency_map.detach().cpu()
        saliency_maps.append(saliency_map)

        if preds_logits[0].numel():
            score = pred_logit
            score.backward()
        # if logits[0].numel():
        #     score = logit[0]
        #     score.backward()

        if nObj != 0:
            self.activations[0] = self.activations[0].detach().cpu()
            self.activations[1] = self.activations[1].detach().cpu()
            self.activations[2] = self.activations[2].detach().cpu()
            pred_logit = pred_logit.detach().cpu()
            logit = logit.detach().cpu()
            class_prob = class_prob.detach().cpu()
            activations = activations.detach().cpu()
            gradients = gradients.detach().cpu()

        gc.collect()
        torch.cuda.empty_cache()

        FrameStack = np.empty((len(raw_data_rec),), dtype=np.object)
        for i in range(len(raw_data_rec)):
            FrameStack[i] = raw_data_rec[i]

        return saliency_maps, pred_list, class_prob_list, head_num_list, FrameStack

    def __call__(self, input_img, img, pred_label):

        return self.forward(input_img, img, pred_label)
