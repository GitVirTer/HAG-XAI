import time
import torch
import torch.nn.functional as F
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


class YOLOV5EigenCAM:

    def __init__(self, model, layer_name, img_size=(640, 640)):
        self.model = model
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        target_layer = find_yolo_layer(self.model, layer_name)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))
        print('[INFO] saliency_map size :', self.activations['value'].shape[2:])

    def forward(self, input_img, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        #gradients = []
        nObj = 0
        b, c, h, w = input_img.size()
        tic = time.time()
        preds, logits = self.model(input_img)
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        # for logit, cls, cls_name, obj_logit in zip(logits[0], preds[1][0], preds[2][0], preds[3][0]):
        #     if class_idx:
        #         #score = obj_logit
        #         score = logit[cls]
        #     else:
        #         score = logit.max()
        #     self.model.zero_grad()
        #     tic = time.time()
        #     score.backward(retain_graph=True)
        #     print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
        #     nObj = nObj + 1
        #     if nObj == 1:
        #         gradients = self.gradients['value']
        #     else:
        #         gradients = gradients + self.gradients['value']
        # gradients = gradients / nObj

        activations = self.activations['value']

        #projections = []
        activations_nda = activations.detach().numpy()

        for activation in activations_nda:
            reshaped_activations = (activation).reshape(
                activation.shape[0], -1).transpose()
            # Centering before the SVD seems to be important here,
            # Otherwise the image returned is negative
            reshaped_activations = reshaped_activations - \
                                   reshaped_activations.mean(axis=0)
            U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
            projection = reshaped_activations @ VT[0, :]
            projection = projection.reshape(activation.shape[1:])
            #projections.append(projection)

        projection = torch.from_numpy(projection)
        saliency_map = torch.unsqueeze(projection, 0)
        saliency_map = torch.unsqueeze(saliency_map, 0)

        #b, k, u, v = gradients.size()
        #alpha = gradients.view(b, k, -1).mean(2)
        #weights = alpha.view(b, k, 1, 1)
        #saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        saliency_maps.append(saliency_map)
        return saliency_maps, logits, preds

    def __call__(self, input_img):
        return self.forward(input_img)
