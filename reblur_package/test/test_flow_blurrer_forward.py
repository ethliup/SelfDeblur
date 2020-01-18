import os
import time
import torch
import argparse
import numpy as np

import cv2
from skimage import io

from reblur_package import FlowBlurrer

def main():
    im1 = io.imread('data/0.png')
    im1 = im1.transpose(2, 0, 1)
    im1 = torch.from_numpy(im1).cuda().unsqueeze(0).float()
    
    im2 = io.imread('data/10.png')
    im2 = im2.transpose(2, 0, 1)
    im2 = torch.from_numpy(im2).cuda().unsqueeze(0).float()
    
    im = torch.cat([im1, im2], dim=0)[:,:3,:,:]

    B, C, H, W = im.size()

    flow = torch.ones([2, H, W]).float().cuda().unsqueeze(0) * (3.0)
    flow[:, 0, :, :] = flow[:, 0, :, :] * 2
    flow = flow.repeat(B, 1, 1, 1)

    im = im / 255.
    flow_blurrer = FlowBlurrer.create_with_implicit_mesh(B, C, H, W, 30)
    forward_start_time = time.time()
    blur_image, mask = flow_blurrer(im, flow)
    print("forward consumes", time.time() - forward_start_time)

    cv2.imshow('image', blur_image.detach().cpu().numpy()[1].transpose(1, 2, 0))
    cv2.imshow('mask', mask.detach().cpu().numpy()[1].transpose(1, 2, 0))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
