import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from reblur_package_lib import Flow_forward_warp_mask
from .utils import generate_2D_mesh

class FlowWarpMaskFunction(Function):
    @staticmethod
    def forward(ctx, forward_warp_mask, input_flow):
        B,_,H,W = input_flow.size()
        output_mask = torch.zeros([B, 1, H, W], dtype=torch.float32, device=input_flow.device)
        forward_warp_mask.get_mask(input_flow, output_mask)
        output_mask = torch.clamp(output_mask, 0., 1.0)
        output_mask = output_mask > 1e-5
        return output_mask

    @staticmethod
    def backward(ctx, grad_mask):
        return None, None

class FlowWarpMask(nn.Module):
    def __init__(self, grid):
        super(FlowWarpMask, self).__init__()
        self.forward_warp_mask = Flow_forward_warp_mask(grid)

    @classmethod
    def create_with_implicit_mesh(cls, B, C, H, W):
        grid, faces = generate_2D_mesh(H,W)
        grid = grid.int().cuda().unsqueeze(0)
        grid = grid.repeat(B, 1, 1, 1)
        return cls(grid)

    def forward(self, input_flow):
        out_mask = FlowWarpMaskFunction.apply(self.forward_warp_mask, input_flow)
        return out_mask