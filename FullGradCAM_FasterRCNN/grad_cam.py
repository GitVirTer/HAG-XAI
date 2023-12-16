# -*- coding: utf-8 -*-
"""
 @File    : grad_cam.py
 @Time    : 2020/3/14 下午4:06
 @Author  : yizuotian
 @Description    :
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import utils_previous as ut

class GradCAM:
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name, class_names, sel_norm_str, sel_method):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
        self.class_names = class_names
        self.sel_norm_str = sel_norm_str
        self.sel_XAImethod = sel_method


    def _get_features_hook(self, module, input, output):
        self.feature = output    #self.feature = output
        print("feature shape:{}".format(self.feature.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    # def __call__(self, inputs, index=0):
    def forward(self, inputs, index=0):

        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        output = self.net.inference([inputs])
        saliency_maps = []
        class_prob_list = []
        head_num_list = []
        raw_data_rec = []
        nObj = 0
        c, h, w = inputs['image'].size()
        pred_list = []
        pred_list.append([])
        pred_list.append([])
        pred_list.append([])
        for output_score, proposal_idx, box_corr, class_id in zip(output[0]['instances'].scores, output[0]['instances'].indices, output[0]['instances'].pred_boxes.tensor, output[0]['instances'].pred_classes):
            if self.class_names[class_id] == 'car' or self.class_names[class_id] == 'bus' or self.class_names[class_id] == 'truck':
            # if self.class_names[class_id] == 'person' or self.class_names[class_id] == 'rider':
                print(output)
                score = output_score                                        #output[0]['instances'].scores[index]
                # proposal_idx = output[0]['instances'].indices[index]  # box来自第几个proposal
                self.net.zero_grad()
                score.backward(retain_graph=True)

                class_prob_score = score
                class_prob_score = torch.unsqueeze(class_prob_score, 0)
                class_prob_list.append(class_prob_score)
                box_corr_t = box_corr[[1,0,3,2]]    #xyxy->yxyx
                # box_corr_t = box_corr
                bbox = box_corr_t.tolist()
                pred_list[0].append([bbox])
                cls = class_id.cpu().data.numpy()
                pred_list[1].append([cls])
                cls_name = self.class_names[class_id]
                pred_list[2].append([cls_name])

                gradients = self.gradient
                activations = self.feature

                if self.sel_XAImethod == 'gradcam':
                    saliency_map = ut.gradcam_operation(activations, gradients)
                elif self.sel_XAImethod == 'gradcampp':
                    saliency_map = ut.gradcampp_operation(activations, gradients)
                elif self.sel_XAImethod == 'fullgradcampp':
                    saliency_map = ut.fullgradcam_operation(activations, gradients)
                elif self.sel_XAImethod == 'fullgradcam':
                    saliency_map = ut.fullgradcamraw_operation(activations, gradients)
                elif self.sel_XAImethod == 'saveRawGradAct':
                    saliency_map, raw_data = ut.saveRawGradAct_operation(activations, gradients)
                    raw_data_rec.append(raw_data)

                # Common
                saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                if self.sel_norm_str == 'norm':
                    saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                    saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

                nObj = nObj + 1
                if nObj == 1:
                    saliency_map_sum = saliency_map
                else:
                    saliency_map_sum = saliency_map_sum + saliency_map

        if nObj == 0:
            saliency_map_sum = torch.zeros([1, 1, h, w])
        else:
            saliency_map_sum = saliency_map_sum / nObj

        # b, k, u, v = gradients.size()
        # alpha = gradients.view(b, k, -1).mean(2)
        # weights = alpha.view(b, k, 1, 1)
        # saliency_map = (weights * activations).sum(1, keepdim=True)
        # saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

        # saliency_map_sum = F.relu(saliency_map_sum)
        saliency_map = saliency_map_sum
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        saliency_map = saliency_map.detach().cpu()
        saliency_maps.append(saliency_map)

        # if output[0]['instances'].scores.numel():
        #     # score = pred_logit
        try:
            score.backward()
        except:
            print('No Score')

        FrameStack = np.empty((len(raw_data_rec),), dtype=np.object)
        for i in range(len(raw_data_rec)):
            FrameStack[i] = raw_data_rec[i]

        return saliency_maps, pred_list, class_prob_list, FrameStack


# class GradCamPlusPlus(GradCAM):
#     def __init__(self, net, layer_name):
#         super(GradCamPlusPlus, self).__init__(net, layer_name)
#
#     def __call__(self, inputs, index=0):
#         """
#
#         :param inputs: {"image": [C,H,W], "height": height, "width": width}
#         :param index: 第几个边框
#         :return:
#         """
#         self.net.zero_grad()
#         output = self.net.inference([inputs])
#         print(output)
#         score = output[0]['instances'].scores[index]
#         proposal_idx = output[0]['instances'].indices[index]  # box来自第几个proposal
#         score.backward()
#
#         gradient = self.gradient[proposal_idx].cpu().data.numpy()  # [C,H,W]
#         gradient = np.maximum(gradient, 0.)  # ReLU
#         indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
#         norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
#         for i in range(len(norm_factor)):
#             norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
#         alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]
#
#         weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)
#
#         feature = self.feature[proposal_idx].cpu().data.numpy()  # [C,H,W]
#
#         cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
#         cam = np.sum(cam, axis=0)  # [H,W]
#         # cam = np.maximum(cam, 0)  # ReLU
#
#         # 数值归一化
#         cam -= np.min(cam)
#         cam /= np.max(cam)
#         # resize to box scale
#         box = output[0]['instances'].pred_boxes.tensor[index].detach().numpy().astype(np.int32)
#         x1, y1, x2, y2 = box
#         cam = cv2.resize(cam, (x2 - x1, y2 - y1))
#
#         return cam

    def __call__(self, input_img):

        return self.forward(input_img)