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
import matplotlib.pyplot as plt
from scipy import ndimage
# from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_filter

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
    res_img = Box.put_box(res_img, bbox, thickness=2)
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

def compute_faith(model, img, masks_ndarray, label_data_corr_xywh, class_names_sel):
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
            if cls_name_one in class_names_sel:
                boxes_rescale_xyxy, boxes_rescale_xywh, _ = rescale_box_list([[bbox_one]], shape_raw, shape_new)
                pred_deletion_adj[0][i].append(boxes_rescale_xyxy.tolist()[0])
                pred_deletion_adj[1][i].append(boxes_rescale_xywh.tolist()[0])
                pred_deletion_adj[2][i].append(cls_idx_one)
                pred_deletion_adj[3][i].append(cls_name_one)
                pred_deletion_adj[4][i].append(conf_one)

    # Show Examples
    # show_idx = [0, 12, 24, 36, 48, 60, 72, 84, 96]
    # img_show_torch = torch_img_deletion_rec[show_idx,:,:,:]
    # img_show_ndarray = img_show_torch.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1, 0).detach().cpu().numpy().astype('uint8')
    # img_show_ndarray_cat1 = np.concatenate((img_show_ndarray[:,:,:,0], img_show_ndarray[:,:,:,1], img_show_ndarray[:,:,:,2]), axis=1)
    # img_show_ndarray_cat2 = np.concatenate((img_show_ndarray[:,:,:,3], img_show_ndarray[:,:,:,4], img_show_ndarray[:,:,:,5]), axis=1)
    # img_show_ndarray_cat3 = np.concatenate((img_show_ndarray[:,:,:,6], img_show_ndarray[:,:,:,7], img_show_ndarray[:,:,:,8]), axis=1)
    # img_show_ndarray_cat = np.concatenate((img_show_ndarray_cat1, img_show_ndarray_cat2, img_show_ndarray_cat3), axis=0)
    # plt.figure()
    # plt.imshow(img_show_ndarray_cat)
    # plt.show()

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
            if cls_name_one in class_names_sel:
                boxes_rescale_xyxy, boxes_rescale_xywh, _ = rescale_box_list([[bbox_one]], shape_raw, shape_new)
                pred_insertation_adj[0][i].append(boxes_rescale_xyxy.tolist()[0])
                pred_insertation_adj[1][i].append(boxes_rescale_xywh.tolist()[0])
                pred_insertation_adj[2][i].append(cls_idx_one)
                pred_insertation_adj[3][i].append(cls_name_one)
                pred_insertation_adj[4][i].append(conf_one)

    # Show Examples
    # show_idx = [0, 12, 24, 36, 48, 60, 72, 84, 99]
    # img_show_torch = torch_img_insertation_rec[show_idx,:,:,:]
    # img_show_ndarray = img_show_torch.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1, 0).detach().cpu().numpy().astype('uint8')
    # img_show_ndarray_cat1 = np.concatenate((img_show_ndarray[:,:,:,0], img_show_ndarray[:,:,:,1], img_show_ndarray[:,:,:,2]), axis=1)
    # img_show_ndarray_cat2 = np.concatenate((img_show_ndarray[:,:,:,3], img_show_ndarray[:,:,:,4], img_show_ndarray[:,:,:,5]), axis=1)
    # img_show_ndarray_cat3 = np.concatenate((img_show_ndarray[:,:,:,6], img_show_ndarray[:,:,:,7], img_show_ndarray[:,:,:,8]), axis=1)
    # img_show_ndarray_cat = np.concatenate((img_show_ndarray_cat1, img_show_ndarray_cat2, img_show_ndarray_cat3), axis=0)
    # plt.figure()
    # plt.imshow(img_show_ndarray_cat)
    # plt.show()

    return pred_deletion_adj, pred_insertation_adj, thr_descend

def compute_faith_specificBBox(model, img, masks_ndarray, label_data_corr_xywh):
    # Compute Region Area
    valid_area = 0
    for cor in label_data_corr_xywh:
        valid_area = valid_area + cor[2]*cor[3]
    valid_area = np.round(1*valid_area).astype('int32')

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
    img_raw = img.copy()
    img_raw_float = img_raw.astype('float')/255
    device = 'cuda' if next(model.model.parameters()).is_cuda else 'cpu'
    torch_img_deletion_rec = torch.zeros(0, torch_img.size(1), torch_img.size(2), torch_img.size(3), device=device)
    for i_thr in thr_descend:
        img_raw_float_use = img_raw_float.copy()
        # img_raw_float_use[masks_ndarray_RGB >= i_thr] = img_raw_float_use.mean()    #np.random.rand(sum(sum(sum(masks_ndarray_RGB >= i_thr))), )
        img_raw_float_use[masks_ndarray_RGB >= i_thr] = np.random.rand(sum(sum(sum(masks_ndarray_RGB >= i_thr))), )
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
    torch_img_deletion_rec = F.upsample(torch_img_deletion_rec, size=(np.size(img, 0), np.size(img, 1)), mode='bilinear', align_corners=False)
    show_idx = [0, 12, 24, 36, 48, 60, 72, 84, 96]
    img_show_torch = torch_img_deletion_rec[show_idx,:,:,:]
    img_show_ndarray = img_show_torch.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1, 0).detach().cpu().numpy().astype('uint8')
    color_order = [2,1,0]
    img_show_ndarray_cat1 = np.concatenate((img_show_ndarray[:,:,color_order,0], img_show_ndarray[:,:,color_order,1], img_show_ndarray[:,:,color_order,2]), axis=1)
    img_show_ndarray_cat2 = np.concatenate((img_show_ndarray[:,:,color_order,3], img_show_ndarray[:,:,color_order,4], img_show_ndarray[:,:,color_order,5]), axis=1)
    img_show_ndarray_cat3 = np.concatenate((img_show_ndarray[:,:,color_order,6], img_show_ndarray[:,:,color_order,7], img_show_ndarray[:,:,color_order,8]), axis=1)
    img_show_ndarray_cat = np.concatenate((img_show_ndarray_cat1, img_show_ndarray_cat2, img_show_ndarray_cat3), axis=0)
    # plt.figure()
    # plt.imshow(img_show_ndarray_cat)
    # plt.show()

    img_withcorr = []
    for i, (i_idx) in enumerate(show_idx):
        img_show_corr = pred_deletion_adj[0][i_idx]
        gt_img = img_show_ndarray[:,:,:,i].copy()
        gt_img = gt_img / gt_img.max()
        gt_img = gt_img[..., ::-1]
        for bbox in img_show_corr:
            bbox_array = np.array(bbox)[[1,0,3,2]].tolist()
            gt_img = put_text_box(bbox_array, ' ', gt_img) / 255
        final_image = concat_images([gt_img * 255])
        img_withcorr.append(final_image)

    img_show_ndarray_cat1 = np.concatenate((img_withcorr[0], img_withcorr[1], img_withcorr[2]), axis=1)
    img_show_ndarray_cat2 = np.concatenate((img_withcorr[3], img_withcorr[4], img_withcorr[5]), axis=1)
    img_show_ndarray_cat3 = np.concatenate((img_withcorr[6], img_withcorr[7], img_withcorr[8]), axis=1)
    img_show_withcorr_cat = np.concatenate((img_show_ndarray_cat1, img_show_ndarray_cat2, img_show_ndarray_cat3), axis=0)

    return img_show_ndarray_cat, img_show_withcorr_cat

def compute_perturbation_specificBBox(model, img, masks_ndarray, label_data_corr_xywh):
    # Pre-define row col:
    nRow = 4
    nCol = 5
    nPics = nRow * nCol

    ## Porpotional setting
    mask_height = np.round(label_data_corr_xywh[:, 2] / nCol)*3
    mask_width = np.round(label_data_corr_xywh[:, 3] / nRow)*3

    ## Fixed setting
    # mask_width = 10
    # mask_height = 10

    ## Square setting
    # mask_width = min(label_data_corr_xywh[2] / nCol, label_data_corr_xywh[3] / nRow)
    # mask_height = min(label_data_corr_xywh[2] / nCol, label_data_corr_xywh[3] / nRow)

    # Define Xc, Yc
    m_yc = np.linspace(label_data_corr_xywh[:,0]-label_data_corr_xywh[:,2]/2,
                       label_data_corr_xywh[:,0]+label_data_corr_xywh[:,2]/2, nCol)
    m_xc = np.linspace(label_data_corr_xywh[:, 1] - label_data_corr_xywh[:, 3] / 2,
                       label_data_corr_xywh[:, 1] + label_data_corr_xywh[:, 3] / 2, nRow)

    # Generate xywh
    torch_img = model.preprocessing(img[..., ::-1])
    img_raw = img.copy()
    img_raw_float = img_raw.astype('float')/255
    device = 'cuda' if next(model.model.parameters()).is_cuda else 'cpu'
    torch_img_deletion_rec = torch.zeros(0, torch_img.size(1), torch_img.size(2), torch_img.size(3), device=device)
    np_img_deletion_rec = np.zeros((img.shape[0], img.shape[1], img.shape[2], 0), np.uint8)
    for cur_xc in m_xc:
        for cur_yc in m_yc:
            curCorr = np.array([cur_xc, cur_yc, mask_width, mask_height]).round()
            curCorr_xyxy = np.round(xywh2xyxy(curCorr.T)).squeeze()
            curCorr_xyxy[curCorr_xyxy<=0] = 0
            curCorr_xyxy = curCorr_xyxy.astype('uint32')
            if curCorr_xyxy[2] > img.shape[0]:
                curCorr_xyxy[2] = img.shape[0]
            if curCorr_xyxy[3] > img.shape[1]:
                curCorr_xyxy[3] = img.shape[1]
            img_raw_float_use = img_raw_float.copy()
            ## Random Fill
            img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = \
                np.random.rand((curCorr_xyxy[2]-curCorr_xyxy[0]), (curCorr_xyxy[3]-curCorr_xyxy[1]), 3)
            ## Mean Fill
            # img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = img_raw_float_use.mean()
            ## Zero Fill
            # img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = 0

            img_raw_uint8_use = (img_raw_float_use*255).astype('uint8')
            torch_img_rand = model.preprocessing(img_raw_uint8_use[..., ::-1])
            torch_img_deletion_rec = torch.cat((torch_img_deletion_rec, torch_img_rand), 0)

            # Expand the array into a 4-D ndarray by inserting a new axis at position 3
            img_raw_uint8_use = np.expand_dims(img_raw_uint8_use, axis=3)
            # Concatenate the two arrays along the 4th dimension
            np_img_deletion_rec = np.concatenate((np_img_deletion_rec, img_raw_uint8_use), axis=3)

            ## examples
            # plt.figure()
            # plt.imshow(torch_img_rand.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1,0).squeeze().detach().cpu().numpy().astype('uint8'))
            # plt.show()

    with torch.no_grad():
        preds_deletion, logits_deletion, preds_logits_deletion, classHead_output_deletion = model(torch_img_deletion_rec)

    # Predictions
    shape_raw = [torch_img_rand.size(3), torch_img_rand.size(2)]  # w, h
    shape_new = [np.size(img, 1), np.size(img, 0)]  # w, h
    pred_deletion_adj = [[[] for _ in range(nPics)] for _ in range(5)]
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
    torch_img_deletion_rec = F.upsample(torch_img_deletion_rec, size=(np.size(img, 0), np.size(img, 1)), mode='bilinear', align_corners=False)
    # show_idx = [0, 12, 24, 36, 48, 60, 72, 84, 96]
    img_show_torch = torch_img_deletion_rec
    img_show_ndarray = img_show_torch.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1, 0).detach().cpu().numpy().astype('uint8')
    color_order = [2,1,0]

    # USE original img ndarray
    img_show_ndarray = np_img_deletion_rec
    color_order = [0, 1, 2]

    cnt = 0
    img_show_ndarray_cat_list = []
    for cur_xc in m_xc:
        curColData_list = []
        for cur_yc in m_yc:
            curColData_list.append(img_show_ndarray[:,:,color_order,cnt])
            cnt = cnt + 1
        curColData = np.concatenate(curColData_list, axis=1)
        img_show_ndarray_cat_list.append(curColData)
    img_show_ndarray_cat = np.concatenate(img_show_ndarray_cat_list, axis=0)
    # plt.figure()
    # plt.imshow(img_show_ndarray_cat)
    # plt.show()

    # Add text boxes
    img_withcorr = []
    for i in range(0, nPics):
        img_show_corr = pred_deletion_adj[0][i]
        gt_img = img_show_ndarray[:,:,:,i].copy()
        gt_img = gt_img / gt_img.max()
        gt_img = gt_img[..., ::-1]
        for bbox in img_show_corr:
            bbox_array = np.array(bbox)[[1,0,3,2]].tolist()
            gt_img = put_text_box(bbox_array, ' ', gt_img) / 255
        final_image = concat_images([gt_img * 255])
        img_withcorr.append(final_image)

    cnt = 0
    img_show_withcorr_cat_list = []
    for cur_xc in m_xc:
        img_show_ndarray_cat_list = []
        for cur_yc in m_yc:
            img_show_ndarray_cat_list.append(img_withcorr[cnt])
            cnt = cnt + 1
        img_show_ndarray_col_cat = np.concatenate(img_show_ndarray_cat_list, axis=1)
        img_show_withcorr_cat_list.append(img_show_ndarray_col_cat)
    img_show_withcorr_cat = np.concatenate(img_show_withcorr_cat_list, axis=0)

    return img_show_ndarray_cat, img_show_withcorr_cat

def compute_perturbationSmo_specificBBox(model, img, masks_ndarray, label_data_corr_xywh):
    # Define Gaussian Kernel
    size = 31
    sigma = 5
    x, y = np.meshgrid(np.linspace(-(size - 1) / 2, (size - 1) / 2, size),
                       np.linspace(-(size - 1) / 2, (size - 1) / 2, size))
    gK = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Pre-define row col:
    nRow = 4
    nCol = 5
    nPics = nRow * nCol

    ## Porpotional setting
    mask_height = np.round(label_data_corr_xywh[:, 2] / nCol)*3
    mask_width = np.round(label_data_corr_xywh[:, 3] / nRow)*3

    ## Fixed setting
    # mask_width = 10
    # mask_height = 10

    ## Square setting
    # mask_width = min(label_data_corr_xywh[2] / nCol, label_data_corr_xywh[3] / nRow)
    # mask_height = min(label_data_corr_xywh[2] / nCol, label_data_corr_xywh[3] / nRow)

    # Define Xc, Yc
    m_yc = np.linspace(label_data_corr_xywh[:,0]-label_data_corr_xywh[:,2]/2,
                       label_data_corr_xywh[:,0]+label_data_corr_xywh[:,2]/2, nCol)
    m_xc = np.linspace(label_data_corr_xywh[:, 1] - label_data_corr_xywh[:, 3] / 2,
                       label_data_corr_xywh[:, 1] + label_data_corr_xywh[:, 3] / 2, nRow)

    # Generate xywh
    torch_img = model.preprocessing(img[..., ::-1])
    img_raw = img.copy()
    img_raw_float = img_raw.astype('float')/255
    device = 'cuda' if next(model.model.parameters()).is_cuda else 'cpu'
    torch_img_deletion_rec = torch.zeros(0, torch_img.size(1), torch_img.size(2), torch_img.size(3), device=device)
    for cur_xc in m_xc:
        for cur_yc in m_yc:
            curCorr = np.array([cur_xc, cur_yc, mask_width, mask_height]).round()
            curCorr_xyxy = np.round(xywh2xyxy(curCorr.T)).squeeze()
            curCorr_xyxy[curCorr_xyxy<=0] = 0
            curCorr_xyxy = curCorr_xyxy.astype('uint32')
            if curCorr_xyxy[2] > img.shape[0]:
                curCorr_xyxy[2] = img.shape[0]
            if curCorr_xyxy[3] > img.shape[1]:
                curCorr_xyxy[3] = img.shape[1]
            img_raw_float_use = img_raw_float.copy()
            ## Random Fill
            RandomFill = np.random.rand((curCorr_xyxy[2] - curCorr_xyxy[0]), (curCorr_xyxy[3] - curCorr_xyxy[1]), 3)
            wFill = np.ones_like(RandomFill)
            img_raw_float_randfill = np.zeros_like(img_raw_float_use)
            img_raw_float_wfill = np.zeros_like(img_raw_float_use)
            img_raw_float_randfill[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = RandomFill
            img_raw_float_wfill[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = wFill
            img_raw_float_randFillSmo = np.zeros_like(img_raw_float_use)
            for iCh in range(0,3):
                img_raw_float_randFillSmo[:,:,iCh] = ndimage.filters.convolve(img_raw_float_randfill[:,:,iCh], gK)
            img_raw_float_wFillSmo = np.zeros_like(img_raw_float_use)
            for iCh in range(0, 3):
                img_raw_float_wFillSmo[:,:,iCh] = ndimage.filters.convolve(img_raw_float_wfill[:,:,iCh], gK)
            img_raw_float_use = (1-img_raw_float_wFillSmo)*img_raw_float_use + img_raw_float_randFillSmo*img_raw_float_wFillSmo

            # img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = \
            #     np.random.rand((curCorr_xyxy[2]-curCorr_xyxy[0]), (curCorr_xyxy[3]-curCorr_xyxy[1]), 3)
            ## Mean Fill
            # img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = img_raw_float_use.mean()
            ## Zero Fill
            # img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = 0

            img_raw_uint8_use = (img_raw_float_use*255).astype('uint8')
            torch_img_rand = model.preprocessing(img_raw_uint8_use[..., ::-1])
            torch_img_deletion_rec = torch.cat((torch_img_deletion_rec, torch_img_rand), 0)

            ## examples
            # plt.figure()
            # plt.imshow(torch_img_rand.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1,0).squeeze().detach().cpu().numpy().astype('uint8'))
            # plt.show()

    with torch.no_grad():
        preds_deletion, logits_deletion, preds_logits_deletion, classHead_output_deletion = model(torch_img_deletion_rec)

    # Predictions
    shape_raw = [torch_img_rand.size(3), torch_img_rand.size(2)]  # w, h
    shape_new = [np.size(img, 1), np.size(img, 0)]  # w, h
    pred_deletion_adj = [[[] for _ in range(nPics)] for _ in range(5)]
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
    torch_img_deletion_rec = F.upsample(torch_img_deletion_rec, size=(np.size(img, 0), np.size(img, 1)), mode='bilinear', align_corners=False)
    # show_idx = [0, 12, 24, 36, 48, 60, 72, 84, 96]
    img_show_torch = torch_img_deletion_rec
    img_show_ndarray = img_show_torch.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1, 0).detach().cpu().numpy().astype('uint8')
    color_order = [2,1,0]

    cnt = 0
    img_show_ndarray_cat_list = []
    for cur_xc in m_xc:
        curColData_list = []
        for cur_yc in m_yc:
            curColData_list.append(img_show_ndarray[:,:,color_order,cnt])
            cnt = cnt + 1
        curColData = np.concatenate(curColData_list, axis=1)
        img_show_ndarray_cat_list.append(curColData)
    img_show_ndarray_cat = np.concatenate(img_show_ndarray_cat_list, axis=0)
    # plt.figure()
    # plt.imshow(img_show_ndarray_cat)
    # plt.show()

    # Add text boxes
    img_withcorr = []
    for i in range(0, nPics):
        img_show_corr = pred_deletion_adj[0][i]
        gt_img = img_show_ndarray[:,:,:,i].copy()
        gt_img = gt_img / gt_img.max()
        gt_img = gt_img[..., ::-1]
        for bbox in img_show_corr:
            bbox_array = np.array(bbox)[[1,0,3,2]].tolist()
            gt_img = put_text_box(bbox_array, ' ', gt_img) / 255
        final_image = concat_images([gt_img * 255])
        img_withcorr.append(final_image)

    cnt = 0
    img_show_withcorr_cat_list = []
    for cur_xc in m_xc:
        img_show_ndarray_cat_list = []
        for cur_yc in m_yc:
            img_show_ndarray_cat_list.append(img_withcorr[cnt])
            cnt = cnt + 1
        img_show_ndarray_col_cat = np.concatenate(img_show_ndarray_cat_list, axis=1)
        img_show_withcorr_cat_list.append(img_show_ndarray_col_cat)
    img_show_withcorr_cat = np.concatenate(img_show_withcorr_cat_list, axis=0)

    return img_show_ndarray_cat, img_show_withcorr_cat

def compute_perturbationGSmo_specificBBox(model, img, masks_ndarray, label_data_corr_xywh):
    # Pre-define row col:
    nRow = 4
    nCol = 5
    nPics = nRow * nCol

    ## Porpotional setting
    mask_height = np.round(label_data_corr_xywh[:, 2] / nCol)*3
    mask_width = np.round(label_data_corr_xywh[:, 3] / nRow)*3

    ## Fixed setting
    # mask_width = 10
    # mask_height = 10

    ## Square setting
    # mask_width = min(label_data_corr_xywh[2] / nCol, label_data_corr_xywh[3] / nRow)
    # mask_height = min(label_data_corr_xywh[2] / nCol, label_data_corr_xywh[3] / nRow)

    # Define Xc, Yc
    m_yc = np.linspace(label_data_corr_xywh[:,0]-label_data_corr_xywh[:,2]/2,
                       label_data_corr_xywh[:,0]+label_data_corr_xywh[:,2]/2, nCol)
    m_xc = np.linspace(label_data_corr_xywh[:, 1] - label_data_corr_xywh[:, 3] / 2,
                       label_data_corr_xywh[:, 1] + label_data_corr_xywh[:, 3] / 2, nRow)

    # Generate xywh
    torch_img = model.preprocessing(img[..., ::-1])
    img_raw = img.copy()
    img_raw_float = img_raw.astype('float')/255
    device = 'cuda' if next(model.model.parameters()).is_cuda else 'cpu'
    torch_img_deletion_rec = torch.zeros(0, torch_img.size(1), torch_img.size(2), torch_img.size(3), device=device)
    for cur_xc in m_xc:
        for cur_yc in m_yc:
            curCorr = np.array([cur_xc, cur_yc, mask_width, mask_height]).round()
            curCorr_xyxy = np.round(xywh2xyxy(curCorr.T)).squeeze()
            curCorr_xyxy[curCorr_xyxy<=0] = 0
            curCorr_xyxy = curCorr_xyxy.astype('uint32')
            if curCorr_xyxy[2] > img.shape[0]:
                curCorr_xyxy[2] = img.shape[0]
            if curCorr_xyxy[3] > img.shape[1]:
                curCorr_xyxy[3] = img.shape[1]
            img_raw_float_use = img_raw_float.copy()
            ## Random Fill
            RandomFill = np.random.rand((curCorr_xyxy[2] - curCorr_xyxy[0]), (curCorr_xyxy[3] - curCorr_xyxy[1]), 3)
            RandomFill = 0.5 + 0*RandomFill

            wFill = np.ones_like(RandomFill)
            img_raw_float_randfill = np.zeros_like(img_raw_float_use)
            img_raw_float_wfill = np.zeros_like(img_raw_float_use)
            img_raw_float_randfill[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = RandomFill
            img_raw_float_wfill[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = wFill

            img_raw_float_randFillSmo = np.zeros_like(img_raw_float_use)
            for iCh in range(0, 3):
                img_raw_float_randFillSmo[:,:,iCh] = gaussian_filter(img_raw_float_randfill[:,:,iCh], sigma=13)
            img_raw_float_wFillSmo = np.zeros_like(img_raw_float_use)
            for iCh in range(0, 3):
                img_raw_float_wFillSmo[:,:,iCh] = gaussian_filter(img_raw_float_wfill[:,:,iCh], sigma=13)

            img_raw_float_use = (1-img_raw_float_wFillSmo)*img_raw_float_use + img_raw_float_randFillSmo*img_raw_float_wFillSmo

            # img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = \
            #     np.random.rand((curCorr_xyxy[2]-curCorr_xyxy[0]), (curCorr_xyxy[3]-curCorr_xyxy[1]), 3)
            ## Mean Fill
            # img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = img_raw_float_use.mean()
            ## Zero Fill
            # img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = 0

            img_raw_uint8_use = (img_raw_float_use*255).astype('uint8')
            torch_img_rand = model.preprocessing(img_raw_uint8_use[..., ::-1])
            torch_img_deletion_rec = torch.cat((torch_img_deletion_rec, torch_img_rand), 0)

            ## examples
            # plt.figure()
            # plt.imshow(torch_img_rand.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1,0).squeeze().detach().cpu().numpy().astype('uint8'))
            # plt.show()

    with torch.no_grad():
        preds_deletion, logits_deletion, preds_logits_deletion, classHead_output_deletion = model(torch_img_deletion_rec)

    # Predictions
    shape_raw = [torch_img_rand.size(3), torch_img_rand.size(2)]  # w, h
    shape_new = [np.size(img, 1), np.size(img, 0)]  # w, h
    pred_deletion_adj = [[[] for _ in range(nPics)] for _ in range(5)]
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
    torch_img_deletion_rec = F.upsample(torch_img_deletion_rec, size=(np.size(img, 0), np.size(img, 1)), mode='bilinear', align_corners=False)
    # show_idx = [0, 12, 24, 36, 48, 60, 72, 84, 96]
    img_show_torch = torch_img_deletion_rec
    img_show_ndarray = img_show_torch.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1, 0).detach().cpu().numpy().astype('uint8')
    color_order = [2,1,0]

    cnt = 0
    img_show_ndarray_cat_list = []
    for cur_xc in m_xc:
        curColData_list = []
        for cur_yc in m_yc:
            curColData_list.append(img_show_ndarray[:,:,color_order,cnt])
            cnt = cnt + 1
        curColData = np.concatenate(curColData_list, axis=1)
        img_show_ndarray_cat_list.append(curColData)
    img_show_ndarray_cat = np.concatenate(img_show_ndarray_cat_list, axis=0)
    # plt.figure()
    # plt.imshow(img_show_ndarray_cat)
    # plt.show()

    # Add text boxes
    img_withcorr = []
    for i in range(0, nPics):
        img_show_corr = pred_deletion_adj[0][i]
        gt_img = img_show_ndarray[:,:,:,i].copy()
        gt_img = gt_img / gt_img.max()
        gt_img = gt_img[..., ::-1]
        for bbox in img_show_corr:
            bbox_array = np.array(bbox)[[1,0,3,2]].tolist()
            gt_img = put_text_box(bbox_array, ' ', gt_img) / 255
        final_image = concat_images([gt_img * 255])
        img_withcorr.append(final_image)

    cnt = 0
    img_show_withcorr_cat_list = []
    for cur_xc in m_xc:
        img_show_ndarray_cat_list = []
        for cur_yc in m_yc:
            img_show_ndarray_cat_list.append(img_withcorr[cnt])
            cnt = cnt + 1
        img_show_ndarray_col_cat = np.concatenate(img_show_ndarray_cat_list, axis=1)
        img_show_withcorr_cat_list.append(img_show_ndarray_col_cat)
    img_show_withcorr_cat = np.concatenate(img_show_withcorr_cat_list, axis=0)

    return img_show_ndarray_cat, img_show_withcorr_cat


def compute_perturbationGK_specificBBox(model, img, masks_ndarray, label_data_corr_xywh):
    # Define Gaussian Kernel
    size = 31
    sigma = 5
    x, y = np.meshgrid(np.linspace(-(size - 1) / 2, (size - 1) / 2, size),
                       np.linspace(-(size - 1) / 2, (size - 1) / 2, size))
    gK = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Pre-define row col:
    nRow = 4
    nCol = 5
    nPics = nRow * nCol

    ## Porpotional setting
    mask_height = np.round(label_data_corr_xywh[:, 2] / nCol)*15
    mask_width = np.round(label_data_corr_xywh[:, 3] / nRow)*15

    ## Fixed setting
    # mask_width = 10
    # mask_height = 10

    ## Square setting
    # mask_width = min(label_data_corr_xywh[2] / nCol, label_data_corr_xywh[3] / nRow)
    # mask_height = min(label_data_corr_xywh[2] / nCol, label_data_corr_xywh[3] / nRow)

    # Define Xc, Yc
    m_yc = np.linspace(label_data_corr_xywh[:,0]-label_data_corr_xywh[:,2]/2,
                       label_data_corr_xywh[:,0]+label_data_corr_xywh[:,2]/2, nCol)
    m_xc = np.linspace(label_data_corr_xywh[:, 1] - label_data_corr_xywh[:, 3] / 2,
                       label_data_corr_xywh[:, 1] + label_data_corr_xywh[:, 3] / 2, nRow)

    # Generate xywh
    torch_img = model.preprocessing(img[..., ::-1])
    img_raw = img.copy()
    img_raw_float = img_raw.astype('float')/255
    device = 'cuda' if next(model.model.parameters()).is_cuda else 'cpu'
    torch_img_deletion_rec = torch.zeros(0, torch_img.size(1), torch_img.size(2), torch_img.size(3), device=device)
    for cur_xc in m_xc:
        for cur_yc in m_yc:
            curCorr = np.array([cur_xc, cur_yc, mask_width, mask_height]).round()
            curCorr_xyxy = np.round(xywh2xyxy(curCorr.T)).squeeze()
            curCorr_xyxy[curCorr_xyxy<=0] = 0
            curCorr_xyxy = curCorr_xyxy.astype('uint32')
            if curCorr_xyxy[2] > img.shape[0]:
                curCorr_xyxy[2] = img.shape[0]
            if curCorr_xyxy[3] > img.shape[1]:
                curCorr_xyxy[3] = img.shape[1]
            img_raw_float_use = img_raw_float.copy()
            ## Random Fill
            # img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = \
            #     np.random.rand((curCorr_xyxy[2]-curCorr_xyxy[0]), (curCorr_xyxy[3]-curCorr_xyxy[1]), 3)
            ## Mean Fill
            # img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = img_raw_float_use.mean()
            ## Zero Fill
            # img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = 0
            ## Gaussian Fill
            rectRegionRaw = img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :]
            gK_resized = cv2.resize(gK, dsize=(rectRegionRaw.shape[1], rectRegionRaw.shape[0]), interpolation=cv2.INTER_CUBIC)
            gK_resized = np.expand_dims(gK_resized, 2)
            gK_resized = np.concatenate((gK_resized, gK_resized, gK_resized), 2)
            # img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = (1-gK_resized)*rectRegionRaw + 0.5*gK_resized
            # img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = (1-gK_resized)*rectRegionRaw + np.random.rand((curCorr_xyxy[2]-curCorr_xyxy[0]), (curCorr_xyxy[3]-curCorr_xyxy[1]), 3)*gK_resized
            img_raw_float_use[curCorr_xyxy[0]:curCorr_xyxy[2], curCorr_xyxy[1]:curCorr_xyxy[3], :] = (1-gK_resized)*rectRegionRaw + img_raw_float_use.mean()*gK_resized


            img_raw_uint8_use = (img_raw_float_use*255).astype('uint8')
            torch_img_rand = model.preprocessing(img_raw_uint8_use[..., ::-1])
            torch_img_deletion_rec = torch.cat((torch_img_deletion_rec, torch_img_rand), 0)

            ## examples
            # plt.figure()
            # plt.imshow(torch_img_rand.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1,0).squeeze().detach().cpu().numpy().astype('uint8'))
            # plt.show()

    with torch.no_grad():
        preds_deletion, logits_deletion, preds_logits_deletion, classHead_output_deletion = model(torch_img_deletion_rec)

    # Predictions
    shape_raw = [torch_img_rand.size(3), torch_img_rand.size(2)]  # w, h
    shape_new = [np.size(img, 1), np.size(img, 0)]  # w, h
    pred_deletion_adj = [[[] for _ in range(nPics)] for _ in range(5)]
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
    torch_img_deletion_rec = F.upsample(torch_img_deletion_rec, size=(np.size(img, 0), np.size(img, 1)), mode='bilinear', align_corners=False)
    # show_idx = [0, 12, 24, 36, 48, 60, 72, 84, 96]
    img_show_torch = torch_img_deletion_rec
    img_show_ndarray = img_show_torch.mul(255).add_(0.5).clamp_(0, 255).permute(2, 3, 1, 0).detach().cpu().numpy().astype('uint8')
    color_order = [2,1,0]

    cnt = 0
    img_show_ndarray_cat_list = []
    for cur_xc in m_xc:
        curColData_list = []
        for cur_yc in m_yc:
            curColData_list.append(img_show_ndarray[:,:,color_order,cnt])
            cnt = cnt + 1
        curColData = np.concatenate(curColData_list, axis=1)
        img_show_ndarray_cat_list.append(curColData)
    img_show_ndarray_cat = np.concatenate(img_show_ndarray_cat_list, axis=0)
    # plt.figure()
    # plt.imshow(img_show_ndarray_cat)
    # plt.show()

    # Add text boxes
    img_withcorr = []
    for i in range(0, nPics):
        img_show_corr = pred_deletion_adj[0][i]
        gt_img = img_show_ndarray[:,:,:,i].copy()
        gt_img = gt_img / gt_img.max()
        gt_img = gt_img[..., ::-1]
        for bbox in img_show_corr:
            bbox_array = np.array(bbox)[[1,0,3,2]].tolist()
            gt_img = put_text_box(bbox_array, ' ', gt_img) / 255
        final_image = concat_images([gt_img * 255])
        img_withcorr.append(final_image)

    cnt = 0
    img_show_withcorr_cat_list = []
    for cur_xc in m_xc:
        img_show_ndarray_cat_list = []
        for cur_yc in m_yc:
            img_show_ndarray_cat_list.append(img_withcorr[cnt])
            cnt = cnt + 1
        img_show_ndarray_col_cat = np.concatenate(img_show_ndarray_cat_list, axis=1)
        img_show_withcorr_cat_list.append(img_show_ndarray_col_cat)
    img_show_withcorr_cat = np.concatenate(img_show_withcorr_cat_list, axis=0)

    return img_show_ndarray_cat, img_show_withcorr_cat

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

def load_gt_labels(img, label_path, class_names_gt, class_names_sel):
    label_data = np.loadtxt(label_path, dtype=np.float32, delimiter=' ')
    if len(label_data.shape) == 1:
        label_data = label_data[None,:]
    label_data_class = label_data[:, 0]
    label_data_corr = label_data[:,1:]
    sel_idx = []
    for class_name_sel in class_names_sel:
        sel_idx.append(class_names_gt.index(class_name_sel))
    sel_bbox_idx = []
    label_data_class_names = []
    for i, i_label_data_class in enumerate(label_data_class):
        if i_label_data_class in sel_idx:
            sel_bbox_idx.append(i)
            label_data_class_names.append(class_names_gt[i_label_data_class.astype('int32')])
    label_data_corr = label_data_corr[sel_bbox_idx,:] #filter classes
    label_data_class = label_data_class[sel_bbox_idx] #filter class labels
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

    return boxes_GT, label_data_corr_xyxy, label_data_corr_xywh, label_data_corr_yxyx, label_data_class, label_data_class_names

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

def fullgradcampp_operation(activations, gradients):
    weights = F.relu(gradients)
    saliency_map = (weights * activations).sum(1, keepdim=True)

    saliency_map = F.relu(saliency_map)

    return saliency_map

def fullgradcamneg_operation(activations, gradients):
    weights = F.relu(-gradients)
    saliency_map = (weights * activations).sum(1, keepdim=True)

    saliency_map = F.relu(saliency_map)

    return saliency_map

def fullgradcam_operation(activations, gradients):
    weights = gradients
    saliency_map = (weights * activations).sum(1, keepdim=True)

    saliency_map = F.relu(saliency_map)

    return saliency_map

def saveRawGradAct_operation(activations, gradients):
    weights = F.relu(gradients)
    saliency_map = (weights * activations).sum(1, keepdim=True)
    saliency_map = F.relu(saliency_map)

    np_gradients = gradients.detach().cpu().numpy()
    np_activations = activations.detach().cpu().numpy()

    return saliency_map, [np_gradients, np_activations]






