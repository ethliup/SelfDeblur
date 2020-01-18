import torch
import math
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from correlation_package_lib import corr_cuda

class correlationFunction(Function):
    @staticmethod
    def forward(ctx, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply, input1, input2):
        ctx.pad_size=pad_size
        ctx.kernel_size=kernel_size
        ctx.max_displacement=max_displacement
        ctx.stride1=stride1
        ctx.stride2=stride2
        ctx.corr_multiply=corr_multiply
        ctx.input1=input1
        ctx.input2=input2

        B,C,H,W = input1.size()

        paddedbottomheight = H + 2 * pad_size
        paddedbottomwidth = W + 2 * pad_size

        kernel_radius_ = (kernel_size - 1) / 2
        border_size_ = max_displacement + kernel_radius_
        nOutputCols = int((paddedbottomwidth - border_size_ * 2.) / stride1+0.5)
        nOutputRows = int((paddedbottomheight - border_size_ * 2.) / stride1+0.5)
        neighborhood_grid_radius_ = max_displacement // stride2
        neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1
        nOutputPlane = neighborhood_grid_width_ * neighborhood_grid_width_

        rbot1 = torch.zeros([B, C, paddedbottomheight, paddedbottomwidth], dtype=torch.float32, device=input1.device)
        rbot2 = torch.zeros([B, C, paddedbottomheight, paddedbottomwidth], dtype=torch.float32, device=input1.device)
        output = torch.zeros([B, nOutputPlane, nOutputRows, nOutputCols], dtype=torch.float32, device=input1.device)

        corr_cuda().forward(input1, input2,
                        rbot1, rbot2,
                        output,
                        pad_size,
                        kernel_size,
                        max_displacement,
                        stride1,
                        stride2,
                        corr_multiply,
                        B,
                        C,
                        H,
                        W)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        pad_size=ctx.pad_size
        kernel_size=ctx.kernel_size
        max_displacement=ctx.max_displacement
        stride1=ctx.stride1
        stride2=ctx.stride2
        corr_multiply=ctx.corr_multiply
        input1=ctx.input1
        input2=ctx.input2

        B,C,H,W = input1.size()

        paddedbottomheight = H + 2 * pad_size;
        paddedbottomwidth = W + 2 * pad_size;

        rbot1 = torch.zeros([B, C, paddedbottomheight, paddedbottomwidth], dtype=torch.float32, device=input1.device)
        rbot2 = torch.zeros([B, C, paddedbottomheight, paddedbottomwidth], dtype=torch.float32, device=input1.device)
        
        grad_input1 = torch.zeros_like(input1)
        grad_input2 = torch.zeros_like(input2)

        corr_cuda().backward(input1, input2,
                        rbot1, rbot2,
                        grad_output,
                        grad_input1,
                        grad_input2,
                        pad_size,
                        kernel_size,
                        max_displacement,
                        stride1,
                        stride2,
                        corr_multiply,
                        B,C,H,W)

        return None, None, None, None, None, None, grad_input1, grad_input2

class Correlation(nn.Module):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20,
                 stride1=1, stride2=1, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        return correlationFunction.apply(self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply, input1, input2)






