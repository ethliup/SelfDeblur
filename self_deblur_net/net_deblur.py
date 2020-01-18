import torch
import torch.nn as nn
from net_basics import *

class Deblur_net(nn.Module):
    def __init__(self, n_in, n_init, n_out):
        super(Deblur_net, self).__init__()

        # encoder 
        self.en_conv0=Conv2d(n_in, n_init, False, nn.ReLU(), 7, 1)
        self.en_resB0=Cascaded_resnet_blocks(n_init,3)

        self.en_conv1=Conv2d(n_init, n_init*2, False, nn.ReLU(), 3, 2)
        self.en_resB1=Cascaded_resnet_blocks(n_init*2,3)

        self.en_conv2=Conv2d(n_init*2, n_init*4, False, nn.ReLU(), 3, 2)
        self.en_resB2=Cascaded_resnet_blocks(n_init*4,3)        

        self.en_conv3=Conv2d(n_init*4, n_init*8, False, nn.ReLU(), 3, 2)
        self.en_resB3=Cascaded_resnet_blocks(n_init*8,3)        

        # decoder 
        self.de_resB3=Cascaded_resnet_blocks(n_init*8,3)
        self.de_upconv3=Deconv2d(n_init*8,n_init*4)

        self.de_resB2=Cascaded_resnet_blocks(n_init*4,3)
        self.de_upconv2=Deconv2d(n_init*4,n_init*2) 
        
        self.de_resB1=Cascaded_resnet_blocks(n_init*2,3)
        self.de_upconv1=Deconv2d(n_init*2,n_init)

        self.de_resB0=Cascaded_resnet_blocks(n_init,3)
        self.im0=Conv2d(n_init, n_out, False, None, 5, 1)

    def forward(self, im_blur):
        # encode image
        en_feat0=self.en_resB0(self.en_conv0(im_blur))
        en_feat1=self.en_resB1(self.en_conv1(en_feat0))
        en_feat2=self.en_resB2(self.en_conv2(en_feat1))
        en_feat3=self.en_resB3(self.en_conv3(en_feat2))

        # decode image
        de_feat3=self.de_upconv3(self.de_resB3(en_feat3))
        de_feat2=self.de_upconv2(self.de_resB2(en_feat2+de_feat3))
        de_feat1=self.de_upconv1(self.de_resB1(en_feat1+de_feat2))
        im_sharp=self.im0(self.de_resB0(en_feat0+de_feat1))

        return im_sharp

net=Deblur_net(3, 64, 3)
