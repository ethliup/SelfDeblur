import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from reblur_package_lib import Flow_blurrer_layer
from .utils import generate_2D_mesh

class FlowBlurrerFunction(Function):
    @staticmethod
    def forward(ctx, blurrer, mesh_vertex_texture, mesh_flow_middle_to_last, target_img_H, target_img_W):
        ctx.blurrer = blurrer
        B, C, ctx.mesh_vertex_H, ctx.mesh_vertex_W = mesh_vertex_texture.size()
        
        out_blur_image = torch.zeros([B, C, target_img_H, target_img_W], dtype=torch.float32, device=mesh_vertex_texture.device)
        out_mask = torch.zeros([B, target_img_H, target_img_W], dtype=torch.float32, device=mesh_vertex_texture.device)

        blurrer.forward(mesh_vertex_texture, mesh_flow_middle_to_last, out_blur_image, out_mask)
        return out_blur_image, out_mask.unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_blur_image, grad_mask):
        blurrer = ctx.blurrer
        mesh_vertex_H = ctx.mesh_vertex_H
        mesh_vertex_W = ctx.mesh_vertex_W

        B, C, _, _ = grad_blur_image.size()

        #print('grad_blur_image\n', grad_blur_image[:, :, 32, 32])
        #print('mesh_HW', mesh_vertex_H, mesh_vertex_W)

        grad_texture = torch.zeros([B, C, mesh_vertex_H, mesh_vertex_W], dtype=torch.float32, device=grad_blur_image.device)
        grad_flow =  torch.zeros([B, 2, mesh_vertex_H, mesh_vertex_W], dtype=torch.float32, device=grad_blur_image.device)

        blurrer.backward(grad_blur_image, grad_texture, grad_flow)
        return None, grad_texture, grad_flow, None, None


class FlowBlurrer(nn.Module):
    def __init__(self, grid, face, n_image_channels, image_H, image_W, blur_kernel_size):
        super(FlowBlurrer, self).__init__()
        self.blurrer = Flow_blurrer_layer(grid, face, n_image_channels, image_H, image_W, blur_kernel_size)
        self.tar_image_H = image_H
        self.tar_image_W = image_W

    @classmethod
    def create_with_implicit_mesh(cls, B, C, H, W, blur_kernel_size):
        grid, faces = generate_2D_mesh(H,W)
        grid = grid.int().cuda().unsqueeze(0)
        faces = torch.from_numpy(faces.astype(np.int32)).cuda()
        grid = grid.repeat(B, 1, 1, 1)
        faces = faces.repeat(B, 1, 1)
        return cls(grid, faces, C, H, W, blur_kernel_size)

    def forward(self, mesh_texure, mesh_flow_middle_to_last):
        blur_image, mask = FlowBlurrerFunction.apply(self.blurrer, mesh_texure, mesh_flow_middle_to_last, self.tar_image_H, self.tar_image_W)
        return blur_image, mask
