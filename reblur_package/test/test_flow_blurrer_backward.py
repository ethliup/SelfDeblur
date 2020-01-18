import os
import time
import torch
import numpy as np
from reblur_package import FlowBlurrer

def main():
    H = 48
    W = 64
    Kernel_size = 10
    Batch_size = 1

    grid = torch.zeros((Batch_size, 2, 3, 1)).cuda().int()
    rgb = torch.zeros((Batch_size, 3, 3, 1)).cuda().float()
    flow = torch.zeros((Batch_size, 2, 3, 1)).cuda().float()
    faces = torch.zeros((Batch_size, 1, 3)).cuda().int()

    grid[:, 0, 0, 0] = 0
    grid[:, 1, 0, 0] = 0
    grid[:, 0, 1, 0] = 0
    grid[:, 1, 1, 0] = H-1
    grid[:, 0, 2, 0] = W-1
    grid[:, 1, 2, 0] = H-1

    rgb[:, 0, 0, 0] = 0.2
    rgb[:, 1, 0, 0] = 0.3
    rgb[:, 2, 0, 0] = 0.4

    rgb[:, 0, 1, 0] = 1.
    rgb[:, 1, 1, 0] = 2.
    rgb[:, 2, 1, 0] = 3.

    rgb[:, 0, 2, 0] = 0.5
    rgb[:, 1, 2, 0] = 0.7
    rgb[:, 2, 2, 0] = 0.9

    flow[:, 0, 0, 0] = 4.1
    flow[:, 1, 0, 0] = 4.

    flow[:, 0, 1, 0] = 4.1
    flow[:, 1, 1, 0] = 4.1

    flow[:, 0, 2, 0] = 4.1
    flow[:, 1, 2, 0] = 4.1

    faces[:, 0, 0] = 0
    faces[:, 0, 1] = 1
    faces[:, 0, 2] = 2

    rgb.requires_grad = True
    flow.requires_grad = True

    #rgb = rgb / 255.
    flow_blurrer = FlowBlurrer(grid, faces, 3, H, W, Kernel_size)
    blur_image, mask = flow_blurrer(rgb, flow)
    
    #============ Analytical backward ==============
    error2 = (blur_image[:, :, :, :] - 1)**2
    error2 = error2 * mask.float()
    loss = torch.sum(error2[:,:,:,:])
    loss.backward()

    grad_flow = flow.grad.clone()
    grad_rgb = rgb.grad.clone()

    #============ Numerical backward ===============
    grad2_flow = np.zeros_like(grad_flow)
    grad2_rgb = np.zeros_like(grad_rgb)

    for i in range(Batch_size):
        for j in range(2):
            for k in range(3):
                eps = 1e-2
                flow2 = flow.clone()
                flow2[i, j, k, 0] += eps
                images, mask = FlowBlurrer(grid, faces, 3, H, W, Kernel_size)(rgb, flow2)
                error2 = (images[:, :, :, :] - 1.0)**2
                error2 = error2 * mask.float()
                loss2 = torch.sum(error2[:,:,:,:])
                
                grad = ((loss2 - loss) / eps).item()
                grad2_flow[i, j, k, 0] = grad

    for i in range(Batch_size):
        for j in range(3):
            for k in range(3):
                eps = 1e-3
                rgb2 = rgb.clone()
                rgb2[i, j, k, 0] += eps
                images, _ = FlowBlurrer(grid, faces, 3, H, W, Kernel_size)(rgb2, flow)
                error2 = (images[:, :, :, :] - 1.0)**2
                error2 = error2 * mask.float()
                loss2 = torch.sum(error2)
                grad2_rgb[i, j, k, 0] = ((loss2 - loss) / eps).item()

    print('--------------------------------------------')
    print('-------------Test batch size----------------')
    print('--------------------------------------------')
    print('numerical grad_flow', grad2_flow*1000)
    print('analytical grad_flow', grad_flow*1000)
    print('********************************************')
    print('numerical grad_rgb', grad2_rgb)
    print('analytical grad_rgb', grad_rgb)

    print('--------------------------------------------')
    print('-------------Test repeating call------------')
    print('--------------------------------------------')
    for i in range(2):
        print('======= ', i, ' =======')
        rgb.grad.data.zero_()
        flow.grad.data.zero_()

        blur_image, mask = flow_blurrer(rgb, flow)
        error2 = (blur_image - 1.0)**2
        error2 = error2 * mask.float()
        loss = torch.sum(error2)
        loss.backward()
        print('loss', loss)

        grad3_flow = flow.grad.clone()
        grad3_rgb = rgb.grad.clone()
        print('analytical grad_flow-1', grad_flow*1000)
        print('analytical grad_flow-2', grad3_flow*1000)
        print('********************************************')
        print('analytical grad_rgb-1', grad_rgb)
        print('analytical grad_rgb-2', grad3_rgb)

if __name__ == '__main__':
    main()
