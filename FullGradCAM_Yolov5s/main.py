

import os
import time
import argparse
import numpy as np
from models.xai_method import YOLOV5XAI
# from models.eigencam import YOLOV5EigenCAM
# from models.eigengradcam import YOLOV5EigenGradCAM
# from models.weightedgradcam import YOLOV5WeightedGradCAM
# from models.gradcamplusplus import YOLOV5GradCAMpp
# from models.fullgradcam import YOLOV5FullGradCAM
# from models.fullgradcamsqsq import YOLOV5FullGradCAMsqsq
# from models.fullgradcamraw import YOLOV5FullGradCAMraw
# from models.fullgradcampp import YOLOV5FullGradCAMpp
from models.yolo_v5_object_detector import YOLOV5TorchObjectDetector
import cv2
from deep_utils import Box, split_extension
import scipy.io
import torch
# from numba import cuda
from GPUtil import showUtilization as gpu_usage
import gc
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.image import imread
import math
import util_my_yolov5 as ut

import torch.utils.data
# from torch.utils.tensorboard import SummaryWriter
#
# import test  # import test.py to get mAP after each epoch
# from models.yolo import Model
# from utils import google_utils
from utils.datasets import *
from utils.utils import *

# Hyperparameter Settings
# target_layer_group_list = [['model_17_act', 'model_21_act', 'model_25_act'],                    #F1
#                            ['model_14_act', 'model_19_act', 'model_23_act'],                    #F2
#                            ['model_13_act', 'model_17_act', 'model_21_act'],                    #F3
#                            ['model_10_act', 'model_14_act', 'model_19_act'],                    #F4
#                            ['model_9_act', 'model_13_act', 'model_17_act'],                     #F5
#                            ['model_9_act', 'model_10_act', 'model_14_act'],                     #F6
#                            ['model_9_act', 'model_9_act', 'model_13_act'],                      #F7
#                            ['model_9_act', 'model_9_act', 'model_10_act'],                      #F8
#                            ['model_9_act', 'model_9_act', 'model_9_act'],                       #F9
#                            ['model_8_cv2_act', 'model_8_cv2_act', 'model_8_cv2_act'],           #F10
#                            ['model_7_act', 'model_7_act', 'model_7_act'],                       #F11
#                            ['model_6_act', 'model_6_act', 'model_6_act'],                       #F12
#                            ['model_5_act', 'model_5_act', 'model_5_act'],                       #F13
#                            ['model_4_act', 'model_4_act', 'model_4_act'],                       #F14
#                            ['model_3_act', 'model_3_act', 'model_3_act'],                       #F15
#                            ['model_2_act', 'model_2_act', 'model_2_act'],                       #F16
#                            ['model_1_act', 'model_1_act', 'model_1_act']]                       #F17
# target_layer_group_name_list = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17']

# target_layer_group_list = [['model_17_act', 'model_21_act', 'model_25_act'],                    #F1
#                            ['model_14_act', 'model_19_act', 'model_23_act'],                    #F2
#                            ['model_13_act', 'model_17_act', 'model_21_act'],                    #F3
#                            ['model_10_act', 'model_14_act', 'model_19_act'],                    #F4
#                            ['model_9_act', 'model_13_act', 'model_17_act'],                     #F5
#                            ['model_9_act', 'model_10_act', 'model_14_act'],                     #F6
#                            ['model_9_act', 'model_9_act', 'model_13_act'],                      #F7
#                            ['model_9_act', 'model_9_act', 'model_10_act'],                      #F8
#                            ['model_9_act', 'model_9_act', 'model_9_act'],                       #F9
#                            ['model_8_cv2_act', 'model_8_cv2_act', 'model_8_cv2_act']]
# target_layer_group_name_list = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']

target_layer_group_list = [['model_17_act', 'model_21_act', 'model_25_act']]                       #F1
target_layer_group_name_list = ['F1']

input_main_dir = 'raw_images'   #Veh_id_img orib_veh_id_task_previous orib_veh_id_task0922
input_main_dir_label = 'raw_images_labels'   #Veh_id_label orib_veh_id_task_previous_label orib_veh_id_task0922_label
output_main_dir = 'Yolov5s, FullGradCAM++, Layer1'

sel_method = 'fullgradcampp'  # gradcam, gradcampp, fullgradcam, fullgradcampp, saveRawGradAct, saveRawAllAct
sel_nms = 'NMS'
sel_prob = 'class'
sel_norm = 'norm'
sel_model = 'yolov5sbdd100k300epoch.pt'
sel_model_str = sel_model[:-3]
sel_object = 'vehicle'    # human, vehicle
sel_faith = 'nofaith'     # nofaith, aifaith, humanfaith, aihumanfaith, trainedXAIfaith

if sel_object=='human':
    class_names_sel = ['person', 'rider']
elif sel_object=='vehicle':
    class_names_sel = ['car', 'bus', 'truck']

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




def main(img_path, label_path, model, saliency_method, img_num):
    gc.collect()
    torch.cuda.empty_cache()

    class_names_gt = ['person', 'rider', 'car', 'bus', 'truck']
    img = cv2.imread(img_path)
    torch_img = model.preprocessing(img[..., ::-1])

    tic = time.time()
    masks, [boxes, _, class_names, obj_prob], class_prob_list, head_num_list, raw_data = saliency_method(torch_img)
    print("total time:", round(time.time() - tic, 4))
    result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    result = result[..., ::-1]  # convert to bgr
    images = [result]

    ### New Images
    result_raw = result
    images = [img]
    result = img
    masks[0] = F.upsample(masks[0], size=(np.size(img, 0), np.size(img, 1)), mode='bilinear', align_corners=False)
    masks[0] = masks[0]

    ### Rescale Boxes
    shape_raw = [np.size(result_raw, 1), np.size(result_raw, 0)]  # w, h
    shape_new = [np.size(img, 1), np.size(img, 0)]  # w, h
    boxes_rescale_xyxy, boxes_rescale_xywh, boxes = ut.rescale_box_list(boxes, shape_raw, shape_new)

    ### Load labels
    boxes_GT, label_data_corr_xyxy, label_data_corr_xywh, label_data_corr_yxyx, label_data_class, label_data_class_names\
        = ut.load_gt_labels(img, label_path, class_names_gt, class_names_sel)

    ### Calculate AI Performance
    if len(boxes):
        Vacc = ut.calculate_acc(boxes_rescale_xywh, label_data_corr_xywh)/len(boxes_GT)
    else:
        Vacc = 0

    ### Display
    for i, mask in enumerate(masks):
        res_img = result.copy()
        res_img, heat_map = ut.get_res_img(mask, res_img)
    for i, (bbox, cls_name, obj_logit, class_prob, head_num) in enumerate(zip(boxes, class_names, obj_prob, class_prob_list, head_num_list)):
        if cls_name[0] in class_names_sel:
            #bbox, cls_name = boxes[0][i], class_names[0][i]
            # res_img = put_text_box(bbox, cls_name + ": " + str(obj_logit), res_img) / 255
            res_img = ut.put_text_box(bbox[0], cls_name[0] + ": " + str(obj_logit[0]*100)[:2] + ", " + str(class_prob.cpu().detach().numpy()[0]*100)[:2] + ", " + str(head_num[0])[:1], res_img) / 255

    ## Display Ground Truth
    gt_img = result.copy()
    gt_img = gt_img / gt_img.max()
    for i, (bbox, cls_idx) in enumerate(zip(boxes_GT, label_data_class)):
        cls_idx = np.int8(cls_idx)
        if class_names_gt[cls_idx] in class_names_sel:
            #bbox, cls_name = boxes[0][i], class_names[0][i]
            # res_img = put_text_box(bbox, cls_name + ": " + str(obj_logit), res_img) / 255
            gt_img = ut.put_text_box(bbox[0], class_names_gt[cls_idx], gt_img) / 255

    # images.append(gt_img * 255)
    images = [gt_img * 255]
    images.append(res_img * 255)
    final_image = ut.concat_images(images)
    img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
    output_path = f'{args.output_dir}/{img_name}'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[INFO] Saving the final image at {output_path}')
    cv2.imwrite(output_path, final_image)

    gc.collect()
    torch.cuda.empty_cache()

    masks_ndarray = masks[0].squeeze().detach().cpu().numpy()

    # nofaith, aifaith, humanfaith, aihumanfaith
    if sel_faith == 'nofaith':
        scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                      'boxes_pred_xyxy': boxes_rescale_xyxy,
                                                      'boxes_pred_xywh': boxes_rescale_xywh,
                                                      'boxes_gt_xywh': label_data_corr_xywh,
                                                      'boxes_gt_xyxy': label_data_corr_xyxy,
                                                      'HitRate': Vacc,
                                                      'boxes_pred_conf': obj_prob,
                                                      'boxes_pred_class_names': class_names,
                                                      'class_names_sel': class_names_sel,
                                                      'boxes_gt_classes_names': label_data_class_names,
                                                      'grad_act': raw_data
                                                      })
    elif sel_faith == 'aifaith':
        # AI Saliency Map Computation
        preds_deletion, preds_insertation, _ = ut.compute_faith(model, img, masks_ndarray, label_data_corr_xywh, class_names_sel)
        # Saving
        scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                      'boxes_pred_xyxy': boxes_rescale_xyxy,
                                                      'boxes_pred_xywh': boxes_rescale_xywh,
                                                      'boxes_gt_xywh': label_data_corr_xywh,
                                                      'boxes_gt_xyxy': label_data_corr_xyxy,
                                                      'HitRate': Vacc,
                                                      'preds_deletion': preds_deletion,
                                                      'preds_insertation': preds_insertation,
                                                      'boxes_pred_conf': obj_prob,
                                                      'boxes_pred_class_names': class_names,
                                                      'class_names_sel': class_names_sel,
                                                      'boxes_gt_classes_names': label_data_class_names,
                                                      'grad_act': raw_data
                                                      })
    elif sel_faith == 'humanfaith':
        # Human Saliency Map Loading
        human_saliency_map_path = 'C:/D/HKU_XAI_Project/XAI_Human_compare_results_1/human_saliency_map_veh_new_1/' + img_num + '_GSmo_30.mat'
        human_saliency_map = scipy.io.loadmat(human_saliency_map_path)['output_map_norm']
        human_deletion, human_insertation, _ = ut.compute_faith(model, img, human_saliency_map, label_data_corr_xywh, class_names_sel)
        # Saving
        scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                      'boxes_pred_xyxy': boxes_rescale_xyxy,
                                                      'boxes_pred_xywh': boxes_rescale_xywh,
                                                      'boxes_gt_xywh': label_data_corr_xywh,
                                                      'boxes_gt_xyxy': label_data_corr_xyxy,
                                                      'HitRate': Vacc,
                                                      'human_deletion': human_deletion,
                                                      'human_insertation': human_insertation,
                                                      'boxes_pred_conf': obj_prob,
                                                      'boxes_pred_class_names': class_names,
                                                      'class_names_sel': class_names_sel,
                                                      'boxes_gt_classes_names': label_data_class_names,
                                                      'grad_act': raw_data
                                                      })
    elif sel_faith == 'aihumanfaith':
        # AI Saliency Map Computation
        preds_deletion, preds_insertation, _ = ut.compute_faith(model, img, masks_ndarray, label_data_corr_xywh)
        # Human Saliency Map Loading
        human_saliency_map_path = 'C:/D/HKU_XAI_Project/XAI_Human_compare_results_1/human_saliency_map_1/' + img_num + '_GSmo_30.mat'
        human_saliency_map = scipy.io.loadmat(human_saliency_map_path)['output_map_norm']
        human_deletion, human_insertation, _ = ut.compute_faith(model, img, human_saliency_map, label_data_corr_xywh, class_names_sel)
        # Saving
        scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                      'boxes_pred_xyxy': boxes_rescale_xyxy,
                                                      'boxes_pred_xywh': boxes_rescale_xywh,
                                                      'boxes_gt_xywh': label_data_corr_xywh,
                                                      'boxes_gt_xyxy': label_data_corr_xyxy,
                                                      'HitRate': Vacc,
                                                      'preds_deletion': preds_deletion,
                                                      'preds_insertation': preds_insertation,
                                                      'human_deletion': human_deletion,
                                                      'human_insertation': human_insertation,
                                                      'boxes_pred_conf': obj_prob,
                                                      'boxes_pred_class_names': class_names,
                                                      'class_names_sel': class_names_sel,
                                                      'boxes_gt_classes_names': label_data_class_names,
                                                      'grad_act': raw_data
                                                      })
    elif sel_faith == 'trainedXAIfaith':
        # Human Saliency Map Loading
        # trainedXAI_saliency_map_path = 'E:/HKU/HKU_XAI_Project/Human_Inspired_XAI_Try2/saveRawSaliencyMapData_testSet_GaussianConv/' + img_num + '_trainedSaliencyMap.mat'
        # trainedXAI_saliency_map_path = 'E:/HKU/HKU_XAI_Project/Human_Inspired_XAI_Try2/saveRawSaliencyMapData_testSet_GaussianConv_F10/' + img_num + '_trainedSaliencyMap.mat'
        trainedXAI_saliency_map_path = 'H:/Projects/HKU_XAI_Project/Human_Inspired_XAI_Try2/saveRawSaliencyMapData_testSet_GaussianConv_correct/' + img_num + '_trainedSaliencyMap.mat'

        trainedXAI_saliency_map = scipy.io.loadmat(trainedXAI_saliency_map_path)['PredData_raw']
        trainedXAI_deletion, trainedXAI_insertation, _ = ut.compute_faith(model, img, trainedXAI_saliency_map, label_data_corr_xywh, class_names_sel)
        # Saving
        scipy.io.savemat(output_path + '.mat', mdict={'masks_ndarray': masks_ndarray,
                                                      'boxes_pred_xyxy': boxes_rescale_xyxy,
                                                      'boxes_pred_xywh': boxes_rescale_xywh,
                                                      'boxes_gt_xywh': label_data_corr_xywh,
                                                      'boxes_gt_xyxy': label_data_corr_xyxy,
                                                      'HitRate': Vacc,
                                                      'trainedXAI_deletion': trainedXAI_deletion,
                                                      'trainedXAI_insertation': trainedXAI_insertation,
                                                      'boxes_pred_conf': obj_prob,
                                                      'boxes_pred_class_names': class_names,
                                                      'class_names_sel': class_names_sel,
                                                      'boxes_gt_classes_names': label_data_class_names,
                                                      'grad_act': raw_data
                                                      })

    print(f'[INFO] save mat to: {output_path}')

if __name__ == '__main__':
    device = args.device
    # input_size = (args.img_size, args.img_size)
    input_size = (608, 608)


    print('[INFO] Loading the model')

    model = YOLOV5TorchObjectDetector(args.model_path, sel_nms, sel_prob, device, img_size=input_size,
                                      names=None if args.names is None else args.names.strip().split(","))


    for i, (target_layer_group, target_layer_group_name) in enumerate(zip(target_layer_group_list, target_layer_group_name_list)):
        sub_dir_name = sel_method + '_' + sel_object + '_' + sel_nms + '_' + sel_prob + '_' + target_layer_group_name + '_' + sel_faith + '_' + sel_norm + '_' + sel_model_str + '_' + '1'
        args.output_dir = os.path.join(output_main_dir, sub_dir_name)
        args.target_layer = target_layer_group
        saliency_method = YOLOV5XAI(model=model, layer_names=args.target_layer, sel_prob_str=sel_prob,
                                        sel_norm_str=sel_norm, sel_classes=class_names_sel, sel_XAImethod=args.method, img_size=input_size)

        if os.path.isdir(args.img_path):
            img_list = os.listdir(args.img_path)
            label_list = os.listdir(args.label_path)
            print(img_list)
            for item_img, item_label in zip(img_list, label_list):

                main(os.path.join(args.img_path, item_img), os.path.join(args.label_path, item_label), model, saliency_method, item_img[:-4])

                # del model, saliency_method
                gc.collect()
                torch.cuda.empty_cache()
                gpu_usage()

        else:
            main(args.img_path)
