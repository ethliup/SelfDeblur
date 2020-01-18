import torch
import torch.nn as nn

from image_proc import *

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
    def forward(self, output, target, weight=None, mean=False):
        error = torch.abs(output - target)
        if weight is not None:
            error = error * weight.float()
            if mean!=False:
                return error.sum() / weight.float().sum()
        if mean!=False:
            return error.mean()
        return error.sum()

class VariationLoss(nn.Module):
    def __init__(self, nc, grad_fn=Grid_gradient_central_diff):
        super(VariationLoss, self).__init__()
        self.grad_fn = grad_fn(nc)

    def forward(self, image, weight=None, mean=False):
        dx, dy = self.grad_fn(image)
        variation = dx**2 + dy**2

        if weight is not None:
            variation = variation * weight.float()
            if mean!=False:
                return variation.sum() / weight.sum()
        if mean!=False:
            return variation.mean()
        return variation.sum()