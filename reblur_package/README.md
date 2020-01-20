# Differential reblur package

This repository contains a custom pytorch package that reblur a sharp image with dense optical flow blur kernels as described in "Self-supervised Linear Motion Deblurring"

## To install:
```
python setup install
```

## Usage:
```
from reblur_package import FlowBlurrer

Max_blur_kernel_size_in_pixels=30
B,C,H,W = im_sharp.size()
flow = torch.ones([B, 2, H, W]).float().cuda()

flow_blurrer = FlowBlurrer.create_with_implicit_mesh(B, C, H, W, Max_blur_kernel_size_in_pixels)
im_blur, mask = flow_blurrer(im_sharp, flow)
```

#### Acknowledgement
If you find this implementation useful for your work, please acknowledge it appropriately and cite the paper:
```
@article{LiuRAS2020,
  author = {Peidong Liu and Joel Janai and Marc Pollefeys and Torsten Sattler and Andreas Geiger},
  title = {Self-supervised Linear Motion Deblurring},
  journal = {Robotics and Automation Letters},
  year = {2020}
}
```
