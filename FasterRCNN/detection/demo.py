# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import os

import cv2
import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from grad_cam import GradCAM        #, GradCamPlusPlus
from skimage import io
from torch import nn
from utils_previous import get_res_img, put_text_box, concat_images, calculate_acc, scale_coords_new, xyxy2xywh, xywh2xyxy

import argparse
from deep_utils import Box, split_extension
import scipy.io
# from numba import cuda
from GPUtil import showUtilization as gpu_usage
import gc
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.image import imread
import math

# constants
WINDOW_NAME = "COCO detections"

# target_layer_group_list = ['backbone.res4.5.conv3',
#                               'backbone.res4.0.conv3',
#                               'backbone.res3.3.conv3',
#                               'backbone.res3.0.conv3',
#                               'backbone.res2.2.conv3',
#                               'backbone.res2.0.conv3'],                    #F1
# target_layer_group_list = target_layer_group_list[0]
# target_layer_group_name_list = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']

# target_layer_group_list = ['backbone.res2.2.conv3',
#                               'backbone.res2.0.conv3']                  #F1
# target_layer_group_list = target_layer_group_list[0]
# target_layer_group_name_list = ['F5', 'F6']

# target_layer_group_list = ['backbone.res4.5.conv3',
#                               'backbone.res4.4.conv3',
#                               'backbone.res4.3.conv3',
#                               'backbone.res4.2.conv3',
#                               'backbone.res4.1.conv3',
#                               'backbone.res4.0.conv3'],                    #F1
# target_layer_group_list = target_layer_group_list[0]
# target_layer_group_name_list = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']

# target_layer_group_list = ['backbone.res4.0.conv3'],                    #F1
# target_layer_group_list = target_layer_group_list[0]
# target_layer_group_name_list = ['F6']

# target_layer_group_list = ['backbone.res3.3.conv3',
#                            'backbone.res3.2.conv3'],                    #F1
# target_layer_group_list = target_layer_group_list[0]
# target_layer_group_name_list = ['F7', 'F8']

# target_layer_group_list = ['backbone.res5.2.conv3',
#                               'backbone.res4.5.conv3'],                    #F1
# target_layer_group_list = target_layer_group_list[0]
# target_layer_group_name_list = ['F1', 'F2']

# target_layer_group_list = ['backbone.bottom_up.res3.3.conv3'],
# #F backbone.bottom_up.res5.2.conv3 backbone.bottom_up.res4.5.conv3 backbone.bottom_up.res3.3.conv3
# target_layer_group_list = target_layer_group_list[0]
# target_layer_group_name_list = ['F3']

# target_layer_group_list = ['backbone.res3.3.conv3',
#                           'backbone.res3.2.conv3'],                    #F1
# target_layer_group_list = target_layer_group_list[0]
# target_layer_group_name_list = ['F7', 'F8']

# target_layer_group_list = ['backbone.res3.1.conv3',
#                               'backbone.res3.0.conv3'],                    #F1
# target_layer_group_list = target_layer_group_list[0]
# target_layer_group_name_list = ['F9', 'F10']

target_layer_group_list = ['backbone.res2.2.conv3',
                           'backbone.res2.1.conv3',
                           'backbone.res2.0.conv3']                    #F1
target_layer_group_list = target_layer_group_list[0]
target_layer_group_name_list = ['F11', 'F12', 'F13']

# target_layer_group_list = ['backbone.fpn_output5'],                    #F1 fpn_lateral2 fpn_output5
# target_layer_group_list = target_layer_group_list[0]
# target_layer_group_name_list = ['fpn_output5']


input_main_dir = 'Veh_id_img'   #Veh_id_img
input_main_dir_label = 'Veh_id_label'   #Veh_id_label
output_main_dir = 'Veh_id_img_output_update_FasterRCNNself_cpu'
sel_method = 'fullgradcam'
sel_nms = 'NMS'
sel_prob = 'class'
sel_norm = 'norm'
sel_model = 'FasterRCNN_C4_BDD100K.pt'
sel_model_str = sel_model[:-3]

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default=sel_model, help='Path to the model')
parser.add_argument('--img-path', type=str, default=input_main_dir, help='input image path')
parser.add_argument('--output-dir', type=str, default='sample_EM_idtask_1_output_update_2/GradCAM_NMS_objclass_F0_singleScale_norm_v5s_1', help='output dir')
parser.add_argument('--img-size', type=int, default=608, help="input image size")
parser.add_argument('--target-layer', type=list, default=list(target_layer_group_list[0]),
                    help='The layer hierarchical address to which gradcam will applied,'
                         ' the names should be separated by underline')

parser.add_argument('--method', type=str, default=sel_method, help='gradcam or eigencam or eigengradcam or weightedgradcam or gradcampp or fullgradcam')
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--names', type=str, default=None,
                    help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')
parser.add_argument('--label-path', type=str, default=input_main_dir_label, help='input label path')

args = parser.parse_args()

gc.collect()
torch.cuda.empty_cache()
gpu_usage()

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    cfg.freeze()
    return cfg


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


class GuidedBackPropagation(object):

    def __init__(self, net):
        self.net = net
        for (name, module) in self.net.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(self.backward_hook)
        self.net.eval()

    @classmethod
    def backward_hook(cls, module, grad_in, grad_out):
        """

        :param module:
        :param grad_in: tuple,长度为1
        :param grad_out: tuple,长度为1
        :return: tuple(new_grad_in,)
        """
        return torch.clamp(grad_in[0], min=0.0),

    def __call__(self, inputs, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.net.zero_grad()
        output = self.net.inference([inputs])
        score = output[0]['instances'].scores[index]
        score.backward()

        return inputs['image'].grad  # [3,H,W]


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), heatmap


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network='frcnn', output_dir='./results'):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)


def get_parser(img_path, run_device):
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="C:/D/HKU_XAI_Project/FasterRCNNself_GradCAM_pytorch_master_G1/detection/faster_rcnn_R_50_C4_1x.yaml", #"configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input",
                        default=img_path,
                        help="img_path")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "C:/D/HKU_XAI_Project/FasterRCNNself_GradCAM_pytorch_master_G1/detection/model_FasterRCNN_C4_1x_30epoch.pth", "MODEL.DEVICE", run_device],
        nargs=argparse.REMAINDER,
    )
    return parser

def compute_faith(model, img, masks_ndarray, label_data_corr_xywh):
    # Compute Region Area
    valid_area = 0
    for cor in label_data_corr_xywh:
        valid_area = valid_area + cor[2]*cor[3]
    valid_area = np.round(valid_area).astype('int32')

    torch_img = model.preprocessing(img[..., ::-1])

    ### Deletion
    # Sort Saliency Map
    delta_thr = 0
    num_thr = 100
    masks_ndarray[np.isnan(masks_ndarray)] = 0
    masks_ndarray[masks_ndarray <= delta_thr] = 0
    if sum(sum(masks_ndarray)) == 0:
        masks_ndarray[0, 0] = 1
        masks_ndarray[1, 1] = 0.5
    masks_ndarray_flatten = masks_ndarray.flatten()
    # masks_ndarray_positive = masks_ndarray_flatten[masks_ndarray_flatten > delta_thr]
    masks_ndarray_positive = masks_ndarray_flatten
    masks_ndarray_sort = masks_ndarray_positive
    masks_ndarray_sort.sort()   # ascend
    masks_ndarray_sort = masks_ndarray_sort[::-1]   # descend
    masks_ndarray_sort = masks_ndarray_sort[:valid_area]
    masks_ndarray_RGB = np.expand_dims(masks_ndarray, 2)
    masks_ndarray_RGB = np.concatenate((masks_ndarray_RGB, masks_ndarray_RGB, masks_ndarray_RGB),2)
    # thr_ascend = np.linspace(masks_ndarray_sort[0], masks_ndarray_sort[-1], num_thr, False)
    thr_idx = np.floor(np.linspace(0, masks_ndarray_sort.size, num_thr, False))
    thr_descend = masks_ndarray_sort[thr_idx.astype('int')]
    thr_ascend = thr_descend[::-1]
    img_raw = img
    img_raw_float = img_raw.astype('float')/255
    device = 'cuda' if next(model.model.parameters()).is_cuda else 'cpu'
    torch_img_deletion_rec = torch.zeros(0, torch_img.size(1), torch_img.size(2), torch_img.size(3), device=device)
    for i_thr in thr_descend:
        img_raw_float_use = img_raw_float.copy()
        img_raw_float_use[masks_ndarray_RGB > i_thr] = np.random.rand(sum(sum(sum(masks_ndarray_RGB > i_thr))), )
        img_raw_uint8_use = (img_raw_float_use*255).astype('uint8')
        torch_img_rand = model.preprocessing(img_raw_uint8_use[..., ::-1])
        torch_img_deletion_rec = torch.cat((torch_img_deletion_rec, torch_img_rand), 0)
    with torch.no_grad():
        preds_deletion, logits_deletion, preds_logits_deletion, classHead_output_deletion = model(torch_img_deletion_rec)

    shape_raw = [torch_img_rand.size(3), torch_img_rand.size(2)]  # w, h
    shape_new = [np.size(img, 1), np.size(img, 0)]  # w, h
    pred_deletion_adj = [[[] for _ in range(num_thr)] for _ in range(5)]
    for i, (bbox, cls_idx, cls_name, conf) in enumerate(zip(preds_deletion[0], preds_deletion[1], preds_deletion[2], preds_deletion[3])):
        for j, (bbox_one, cls_idx_one, cls_name_one, conf_one) in enumerate(zip(bbox, cls_idx, cls_name, conf)):
            if cls_name_one == 'car' or cls_name_one == 'truck' or cls_name_one == 'bus':
                boxes_rescale_xyxy, boxes_rescale_xywh, _ = rescale_box_list([[bbox_one]], shape_raw, shape_new)
                pred_deletion_adj[0][i].append(boxes_rescale_xyxy.tolist()[0])
                pred_deletion_adj[1][i].append(boxes_rescale_xywh.tolist()[0])
                pred_deletion_adj[2][i].append(cls_idx_one)
                pred_deletion_adj[3][i].append(cls_name_one)
                pred_deletion_adj[4][i].append(conf_one)

    # Show Examples
    show_idx = [0, 12, 24, 36, 48, 60, 72, 84, 99]
    img_show_torch = torch_img_deletion_rec[show_idx,:,:,:]
    img_show_ndarray = img_show_torch.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1, 0).detach().cpu().numpy().astype('uint8')
    img_show_ndarray_cat1 = np.concatenate((img_show_ndarray[:,:,:,0], img_show_ndarray[:,:,:,1], img_show_ndarray[:,:,:,2]), axis=1)
    img_show_ndarray_cat2 = np.concatenate((img_show_ndarray[:,:,:,3], img_show_ndarray[:,:,:,4], img_show_ndarray[:,:,:,5]), axis=1)
    img_show_ndarray_cat3 = np.concatenate((img_show_ndarray[:,:,:,6], img_show_ndarray[:,:,:,7], img_show_ndarray[:,:,:,8]), axis=1)
    img_show_ndarray_cat = np.concatenate((img_show_ndarray_cat1, img_show_ndarray_cat2, img_show_ndarray_cat3), axis=0)
    plt.figure()
    plt.imshow(img_show_ndarray_cat)
    plt.show()

    ### Insertation
    torch_img_insertation_rec = torch.zeros(0, torch_img.size(1), torch_img.size(2), torch_img.size(3), device=device)
    for i_thr in thr_descend:
        img_raw_float_use = img_raw_float.copy()
        img_raw_float_use[masks_ndarray_RGB <= i_thr] = 0
        img_raw_uint8_use = (img_raw_float_use*255).astype('uint8')
        torch_img_rand = model.preprocessing(img_raw_uint8_use[..., ::-1])
        torch_img_insertation_rec = torch.cat((torch_img_insertation_rec, torch_img_rand), 0)
    with torch.no_grad():
        preds_insertation, logits_insertation, preds_logits_insertation, classHead_output_insertation = model(torch_img_insertation_rec)

    pred_insertation_adj = [[[] for _ in range(num_thr)] for _ in range(5)]
    for i, (bbox, cls_idx, cls_name, conf) in enumerate(zip(preds_insertation[0], preds_insertation[1], preds_insertation[2], preds_insertation[3])):
        for j, (bbox_one, cls_idx_one, cls_name_one, conf_one) in enumerate(zip(bbox, cls_idx, cls_name, conf)):
            if cls_name_one == 'car' or cls_name_one == 'truck' or cls_name_one == 'bus':
                boxes_rescale_xyxy, boxes_rescale_xywh, _ = rescale_box_list([[bbox_one]], shape_raw, shape_new)
                pred_insertation_adj[0][i].append(boxes_rescale_xyxy.tolist()[0])
                pred_insertation_adj[1][i].append(boxes_rescale_xywh.tolist()[0])
                pred_insertation_adj[2][i].append(cls_idx_one)
                pred_insertation_adj[3][i].append(cls_name_one)
                pred_insertation_adj[4][i].append(conf_one)

    # Show Examples
    show_idx = [0, 12, 24, 36, 48, 60, 72, 84, 99]
    img_show_torch = torch_img_insertation_rec[show_idx,:,:,:]
    img_show_ndarray = img_show_torch.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1, 0).detach().cpu().numpy().astype('uint8')
    img_show_ndarray_cat1 = np.concatenate((img_show_ndarray[:,:,:,0], img_show_ndarray[:,:,:,1], img_show_ndarray[:,:,:,2]), axis=1)
    img_show_ndarray_cat2 = np.concatenate((img_show_ndarray[:,:,:,3], img_show_ndarray[:,:,:,4], img_show_ndarray[:,:,:,5]), axis=1)
    img_show_ndarray_cat3 = np.concatenate((img_show_ndarray[:,:,:,6], img_show_ndarray[:,:,:,7], img_show_ndarray[:,:,:,8]), axis=1)
    img_show_ndarray_cat = np.concatenate((img_show_ndarray_cat1, img_show_ndarray_cat2, img_show_ndarray_cat3), axis=0)
    plt.figure()
    plt.imshow(img_show_ndarray_cat)
    plt.show()

    return pred_deletion_adj, pred_insertation_adj, thr_descend

def rescale_box_list(boxes, shape_raw, shape_new):
    if len(boxes):
        boxes_ndarray = np.array(boxes).squeeze(1)
        boxes_ndarray = torch.from_numpy(boxes_ndarray) #yxyx
        # img1_shape = [np.size(result_raw, 1), np.size(result_raw, 0)] #w, h
        # img0_shape = [np.size(img, 1), np.size(img, 0)] #w, h
        boxes_rescale = scale_coords_new(shape_raw, boxes_ndarray.float(), shape_new)
        boxes_rescale = boxes_rescale.round()
        boxes_rescale = torch.unsqueeze(boxes_rescale, 1)
        boxes = boxes_rescale.tolist()
        boxes_rescale_yxyx = boxes_rescale.squeeze(1).numpy()
        boxes_rescale_xyxy = boxes_rescale_yxyx[:, [1,0,3,2]]
        boxes_rescale_xywh = xyxy2xywh(boxes_rescale_xyxy)
    else:
        boxes_rescale_xyxy = 0
        boxes_rescale_xywh = 0

    return boxes_rescale_xyxy, boxes_rescale_xywh, boxes

def main(arguments, img_path, label_path, target_layer_group, model, cfg):
    # sel_norm_str = 'norm'

    class_names_gt = ['person', 'rider', 'car', 'bus', 'truck']

    # setup_logger(name="fvcore")
    # logger = setup_logger()
    # logger.info("Arguments: " + str(arguments))
    #
    cfg = setup_cfg(arguments)
    print(cfg)
    # # 构建模型
    # model = build_model(cfg)
    # # 加载权重
    # checkpointer = DetectionCheckpointer(model)
    # checkpointer.load(cfg.MODEL.WEIGHTS)

    # 加载图像
    path = os.path.expanduser(arguments.input)
    img = read_image(path, format="BGR")
    height, width = img.shape[:2]
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    image = transform_gen.get_transform(img).apply_image(img)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).requires_grad_(True)

    inputs = {"image": image, "height": height, "width": width}

    # 获取类别名称
    meta = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )
    # class_names_all = meta.thing_classes
    class_names_all = class_names_gt

    # Grad-CAM
    # layer_name = get_last_conv_name(model)
    layer_name = target_layer_group
    saliencyMap_method = GradCAM(model, layer_name, class_names_all, sel_norm)
    masks, [boxes, _, class_names], class_prob_list = saliencyMap_method(inputs)  # cam mask
    saliencyMap_method.remove_handlers()


    ### Convert Image
    torch_img = image.unsqueeze(0)
    result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    result = result[..., ::-1]  # convert to bgr

    ### New Images
    result_raw = result
    images = [img]
    result = img
    masks[0] = F.upsample(masks[0], size=(np.size(img, 0), np.size(img, 1)), mode='bilinear', align_corners=False)
    masks[0] = masks[0]

    ### Rescale Boxes
    if len(boxes):
        boxes_ndarray = np.array(boxes).squeeze(1)
        boxes_ndarray = torch.from_numpy(boxes_ndarray) #yxyx
        # img1_shape = [np.size(result_raw, 1), np.size(result_raw, 0)] #w, h
        img1_shape = [np.size(img, 1), np.size(img, 0)] #w, h
        img0_shape = [np.size(img, 1), np.size(img, 0)] #w, h
        boxes_rescale = scale_coords_new(img1_shape, boxes_ndarray.float(), img0_shape, ratio_pad=None)
        boxes_rescale = boxes_rescale.round()
        boxes_rescale = torch.unsqueeze(boxes_rescale, 1)
        boxes = boxes_rescale.tolist()
        boxes_rescale_yxyx = boxes_rescale.squeeze(1).numpy()
        boxes_rescale_xyxy = boxes_rescale_yxyx[:, [1,0,3,2]]
        boxes_rescale_xywh = xyxy2xywh(boxes_rescale_xyxy)
    else:
        boxes_rescale_xyxy = 0
        boxes_rescale_xywh = 0

    # boxes[0][0] = [322, 0.0, 381, 35]
    # boxes[1][0] = [322, 0.0, 381, 35]
    # boxes[0][0] = [576, 0, 100, 200]

    ### Load labels
    label_data = np.loadtxt(label_path, dtype=np.float32, delimiter=' ')
    if len(label_data.shape) == 1:
        label_data = label_data[None,:]
    label_data_class = label_data[:, 0]
    label_data_corr = label_data[:,1:]
    label_data_corr = label_data_corr[label_data_class>1,:] #filter classes
    label_data_class = label_data_class[label_data_class>1] #filter class labels
    img_h, img_w = np.size(img, 0), np.size(img, 1)
    label_data_corr[:, 0] = label_data_corr[:, 0] * img_w
    label_data_corr[:, 1] = label_data_corr[:, 1] * img_h
    label_data_corr[:, 2] = label_data_corr[:, 2] * img_w
    label_data_corr[:, 3] = label_data_corr[:, 3] * img_h
    label_data_corr_xywh = label_data_corr
    label_data_corr = xywh2xyxy(label_data_corr)
    label_data_corr_xyxy = label_data_corr
    label_data_corr_yxyx = label_data_corr[:, [1,0,3,2]]
    label_data_corr_yxyx = np.round(label_data_corr_yxyx)
    boxes_GT = label_data_corr_yxyx[:, None, :].tolist()

    masks_ndarray = masks[0].squeeze().detach().cpu().numpy()

    ### Calculate AI Performance
    if len(boxes):
        Vacc = calculate_acc(boxes_rescale_xywh, label_data_corr_xywh)/len(boxes_GT)
    else:
        Vacc = 0

    ### Display
    for i, mask in enumerate(masks):
        res_img = result.copy()
        res_img, heat_map = get_res_img(mask, res_img)
    for i, (bbox, cls_name, class_prob) in enumerate(zip(boxes, class_names, class_prob_list)):
        if cls_name[0] == 'car' or cls_name[0] == 'bus' or cls_name[0] == 'truck':
            #bbox, cls_name = boxes[0][i], class_names[0][i]
            # res_img = put_text_box(bbox, cls_name + ": " + str(obj_logit), res_img) / 255
            res_img = put_text_box(bbox[0], cls_name[0] +  ", " + str(class_prob.cpu().detach().numpy()[0]*100)[:2], res_img) / 255

    ## Display Ground Truth
    gt_img = result.copy()
    gt_img = gt_img / gt_img.max()
    for i, (bbox, cls_idx) in enumerate(zip(boxes_GT, label_data_class)):
        cls_idx = np.int8(cls_idx)
        if class_names_gt[cls_idx] == 'car' or class_names_gt[cls_idx] == 'bus' or class_names_gt[cls_idx] == 'truck':
            #bbox, cls_name = boxes[0][i], class_names[0][i]
            # res_img = put_text_box(bbox, cls_name + ": " + str(obj_logit), res_img) / 255
            gt_img = put_text_box(bbox[0], class_names_gt[cls_idx], gt_img) / 255

    # images.append(gt_img * 255)
    images = [gt_img * 255]
    images.append(res_img * 255)
    final_image = concat_images(images)
    img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
    output_path = f'{args.output_dir}/{img_name}'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[INFO] Saving the final image at {output_path}')
    cv2.imwrite(output_path, final_image)
    scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                  'boxes_pred_xyxy': boxes_rescale_xyxy,
                                                  'boxes_pred_xywh': boxes_rescale_xywh,
                                                  'boxes_gt_xywh': label_data_corr_xywh,
                                                  'boxes_gt_xyxy': label_data_corr_xyxy,
                                                  'HitRate': Vacc})

    gc.collect()
    torch.cuda.empty_cache()







# if __name__ == "__main__":
#     """
#     Usage:export KMP_DUPLICATE_LIB_OK=TRUE
#     python detection/demo.py --config-file detection/faster_rcnn_R_50_C4.yaml \
#       --input ./examples/pic1.jpg \
#       --opts MODEL.WEIGHTS /Users/yizuotian/pretrained_model/model_final_b1acc2.pkl MODEL.DEVICE cpu
#     """
#     mp.set_start_method("spawn", force=True)
#     arguments = get_parser().parse_args()
#     main(arguments)

if __name__ == '__main__':
    device = args.device

    print('[INFO] Loading the model')

    arguments = get_parser(os.path.join(args.img_path, 'test.jpg'), device).parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(arguments))
    cfg = setup_cfg(arguments)
    print(cfg)
    # 构建模型
    model = build_model(cfg)
    # 加载权重
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    for i, (target_layer_group, target_layer_group_name) in enumerate(zip(target_layer_group_list, target_layer_group_name_list)):
        sub_dir_name = sel_method + '_' + sel_nms + '_' + sel_prob + '_' + target_layer_group_name + '_' + 'singleScale' + '_' + sel_norm + '_' + sel_model_str + '_' + '1'
        args.output_dir = os.path.join(output_main_dir, sub_dir_name)
        args.target_layer = target_layer_group

        if os.path.isdir(args.img_path):
            img_list = os.listdir(args.img_path)
            label_list = os.listdir(args.label_path)
            print(img_list)
            for item_img, item_label in zip(img_list, label_list):
                try:
                    arguments = get_parser(os.path.join(args.img_path, item_img), device).parse_args()
                    main(arguments, os.path.join(args.img_path, item_img), os.path.join(args.label_path, item_label), target_layer_group, model, cfg)
                except:
                    arguments = get_parser(os.path.join(args.img_path, item_img), "cpu").parse_args()
                    main(arguments, os.path.join(args.img_path, item_img), os.path.join(args.label_path, item_label), target_layer_group, model, cfg)

                # del model, saliency_method
                gc.collect()
                torch.cuda.empty_cache()
                gpu_usage()

        else:
            main(args.img_path)
