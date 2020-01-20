# Differentiable correlation package

This repository contains a custom pytorch package that adds a module and functional interface for the correlation layer described in "FlowNet: Learning Optical Flow with Convolutional Networks" (https://arxiv.org/abs/1504.06852)

## To install:
```
python setup install
```

## Usages:
```
from correlation_package import Correlation

A = Variable(torch.randn(2,3,100,100), requires_grad=True)
A_ = A.cuda().float()
B = Variable(torch.randn(2,3,100,100), requires_grad=True)
B_ = B.cuda().float()

corr_AB = Correlation(pad_size=3,
                      kernel_size=3,
                      max_displacement=20,
                      stride1=1,
                      stride2=1,
                      corr_multiply=1)(A_, B_)
```

#### Acknowledgement
- We improve the original repository from Dr. Jinwei Gu for better structure and to support Pytorch versions which are higher than 0.4.0
- Thanks to Dr. Fitsum Reda for providing the wrapper to the correlation code

If you find this implementation useful in your work, please acknowledge it appropriately and cite the paper:
```
@misc{flownet2-pytorch,
  author = {Fitsum Reda and Robert Pottorff and Jon Barker and Bryan Catanzaro},
  title = {flownet2-pytorch: Pytorch implementation of FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NVIDIA/flownet2-pytorch}}
}
```
