import torch
import torch.nn as nn

from model_base import *
from net_deblur import *
from metrics import *

class ModelDeblurNet(ModelBase):
    def __init__(self, opts):
        super(ModelDeblurNet, self).__init__()
        self.opts = opts
        
        # create network
        self.model_names=['G']
        self.net_G=Deblur_net(n_in=opts.n_channels, n_init=opts.n_init_feat, n_out=opts.n_channels).cuda()

        self.print_networks(self.net_G)

        if not opts.is_training or opts.continue_train:
            self.load_checkpoint(opts.model_label)

        if opts.is_training: 
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opts.lr)
            self.optimizer_names = ['G']
            self.build_lr_scheduler()
            
            self.loss_fn=nn.L1Loss()
                        
    def set_input(self, _input):
        im_blur, im_target = _input
        self.im_blur=im_blur.cuda()
        if im_target is not None:
            self.im_target=im_target.cuda()
        
    def forward(self):
        im_pred=self.net_G(self.im_blur)
        return im_pred

    def optimize_parameters(self):
        self.im_pred=self.forward()

        self.loss_G=self.loss_fn(self.im_pred, self.im_target)

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()  

    def save_checkpoint(self, label):
        self.save_network(self.net_G, 'deblur', label, self.opts.log_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_G, 'deblur', label, self.opts.log_dir)

    def get_current_scalars(self):
        losses = {}
        losses['loss_G']=self.loss_G.item()
        losses['PSNR_train']=PSNR(self.im_pred.data, self.im_target)
        return losses

    def get_current_visuals(self):
        output_visuals = {}
        output_visuals['im_blur']=self.im_blur
        output_visuals['im_target']=self.im_target
        output_visuals['im_pred']=self.im_pred
        return output_visuals
