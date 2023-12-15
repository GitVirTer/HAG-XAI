from deep_utils import Box, split_extension
import scipy.io
import torch
# from numba import cuda
from GPUtil import showUtilization as gpu_usage
import gc
import torch.nn.functional as F
import cv2
import numpy as np
import math

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def get_res_img(mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
        np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap = (heatmap/255).astype(np.float32)
    #n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    res_img = (res_img / 255).astype(np.float32)
    res_img = cv2.add(res_img, heatmap)
    res_img = (res_img / res_img.max())
    return res_img, heatmap


def put_text_box(bbox, cls_name, res_img):
    x1, y1, x2, y2 = bbox
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    cv2.imwrite('temp.jpg', (res_img * 255).astype(np.uint8))
    res_img = cv2.imread('temp.jpg')
    res_img = Box.put_box(res_img, bbox)
    res_img = Box.put_text(res_img, cls_name, (x1, y1))
    return res_img


def concat_images(images):
    w, h = images[0].shape[:2]
    width = w
    height = h * len(images)
    base_img = np.zeros((width, height, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        base_img[:, h * i:h * (i + 1), ...] = img
    return base_img

def cart2pol(x, y):
    rho = math.sqrt(x ** 2 + y ** 2)
    return (rho)

# added: True if the click is a hit
def hit(xPre, yPre, xloc, yloc, w, h):
    if xPre > xloc - w/2 and xPre < xloc +  w/2 and  yPre > yloc - h/2 and yPre < yloc + h/2:
        return 1
    else:
           return 0
def calculate_acc(AIpre_centres, gt_boxs):
    """RETURN:[0,0,0]"""  # The first line of the method is used to set the return type of the method.
    dist_list = list()
    rect_all = list()
    min_all = list()
    tot_hits = 0
    min_val_all = list()

    hit_idx = list()
    unique_hit_idx_list = list()

    for m in AIpre_centres:
        rho_list = list()
        rect_list = list()

        for t in gt_boxs:
            # print(gt_boxs)
            rho = int(cart2pol(m[0] - t[0], m[1] - t[1]))
            rho_list.append(rho)
            # print(t[0], t[1], t[2], t[3])

            in_rect = hit(m[0], m[1], t[0], t[1], t[2], t[3])
            rect_list.append(in_rect)

        min_val = min(rho_list)
        min_ind = rho_list.index(min_val)

        if sum(rect_list) > 0:
            tot_hits = tot_hits + 1
            hit_idx.append(min_ind)

        dist_list.append(rho_list)
        rect_all.append(rect_list)
        min_all.append(min_ind + 1)
        min_val_all.append(min_val)

    true_tot_hits = 0
    for i in hit_idx:
        if i not in unique_hit_idx_list:
            unique_hit_idx_list.append(i)
            true_tot_hits = true_tot_hits + 1
    tot_hits = true_tot_hits
    print(tot_hits)
    return tot_hits

def scale_coords_new(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] = coords[:, [0, 2]] - pad[0]  # x padding
    coords[:, [1, 3]] = coords[:, [1, 3]] - pad[1]  # y padding
    coords[:, :4] = coords[:, :4] / gain
    clip_coords(coords, img0_shape)
    return coords

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def gradcam_operation(activations, gradients):
    b, k, u, v = gradients.size()
    alpha = gradients.view(b, k, -1).mean(2)
    weights = alpha.view(b, k, 1, 1)

    saliency_map = (weights * activations).sum(1, keepdim=True)
    # saliency_map = (gradients * activations).sum(1, keepdim=True)

    saliency_map = F.relu(saliency_map)

    return saliency_map

def gradcampp_operation(activations, gradients):
    b, k, u, v = gradients.size()
    # alpha = gradients.view(b, k, -1).mean(2)
    # weights = alpha.view(b, k, 1, 1)
    grad = gradients
    features = activations
    alpha_num = grad.pow(2)
    alpha_denom = 2 * alpha_num + features.mul(grad.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    alpha = alpha_num.div(alpha_denom)
    positive_gradients = F.relu(grad)
    weights = alpha * positive_gradients
    weights = weights.sum([2, 3], keepdim=True)

    saliency_map = (weights * activations).sum(1, keepdim=True)

    saliency_map = F.relu(saliency_map)

    return saliency_map

def fullgradcam_operation(activations, gradients):
    weights = F.relu(gradients)
    saliency_map = (weights * activations).sum(1, keepdim=True)

    saliency_map = F.relu(saliency_map)

    return saliency_map

def fullgradcamraw_operation(activations, gradients):
    weights = gradients
    saliency_map = (weights * activations).sum(1, keepdim=True)

    saliency_map = F.relu(saliency_map)

    return saliency_map

def saveRawGradAct_operation(activations, gradients):
    if gradients == None:
        gradients = torch.zeros_like(activations)
    weights = F.relu(gradients)
    saliency_map = (weights * activations).sum(1, keepdim=True)
    saliency_map = F.relu(saliency_map)

    np_gradients = gradients.detach().cpu().numpy()
    np_activations = activations.detach().cpu().numpy()

    return saliency_map, [np_gradients, np_activations]




