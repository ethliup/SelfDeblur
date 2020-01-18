import torch
import torch.nn as nn

## 2D convolution layers
class Conv2d(nn.Module):
    def __init__(self, n_in, n_out, bn, act_fn, ker_sz=3, strd=1):
        super(Conv2d, self).__init__()

        use_bias=True if bn==False else False

        modules = []   
        modules.append(nn.Conv2d(n_in, n_out, ker_sz, stride=strd, padding=(ker_sz-1)//2, bias=use_bias)) 

        if bn==True:
            modules.append(nn.BatchNorm2d(n_out))
        
        if act_fn is not None:
            modules.append(act_fn)

        self.net=nn.Sequential(*modules)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        return self.net(x)

class Deconv2d(nn.Module):
    def __init__(self, n_in, n_out):
        super(Deconv2d, self).__init__()

        self.upsample=nn.Upsample(scale_factor=2, mode="bilinear")
        self.net=nn.Sequential(Conv2d(n_in, n_out, False, None, 3, 1))

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.net(self.upsample(x))

class Resnet_block(nn.Module):
    def __init__(self, n_in):
        super(Resnet_block, self).__init__()

        self.conv_block = self.build_conv_block(n_in)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def build_conv_block(self, n_in):
        conv_block=[]
        conv_block+=[Conv2d(n_in, n_in, False, nn.ReLU(), 3, 1)]
        conv_block+=[Conv2d(n_in, n_in, False, None, 3, 1)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x+self.conv_block(x)

class Cascaded_resnet_blocks(nn.Module):
    def __init__(self, n_in, n_blks):
        super(Cascaded_resnet_blocks, self).__init__()

        resnet_blocks=[]
        for i in range(n_blks):
            resnet_blocks+=[Resnet_block(n_in)]
        self.net = nn.Sequential(*resnet_blocks)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.net(x)
