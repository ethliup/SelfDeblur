import cv2
import torch
from skimage import io

from flow_utils import *
from reblur_package import *

flow = load_flow('data/11439_flow.flo')
flow_rgb = flow2rgb(flow)

prev_image = io.imread('data/11439_img1.ppm')
curr_image = io.imread('data/11439_img2.ppm')

H,W,C=prev_image.shape
mask_fn = FlowWarpMask.create_with_implicit_mesh(1,C,H,W)
flow = torch.from_numpy(flow.transpose(2,0,1)).unsqueeze(0).float().cuda()
mask = mask_fn(flow)

cv2.imshow('flow', flow_rgb)
cv2.imshow('prev_image', prev_image)
cv2.imshow('curr_image', curr_image)
cv2.imshow('mask', mask.float().detach().cpu().numpy()[0].transpose(1, 2, 0))
cv2.waitKey(0)

