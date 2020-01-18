import torch
import numpy as np
import torch.nn as nn

from model_base import *
from net_deblur import *
from net_pwc import*
from reblur_package import *
from flow_utils import *
from image_proc import *
from losses import *

class ModelSelfDeblurNet(ModelBase):
    def __init__(self, opts):
        super(ModelSelfDeblurNet, self).__init__()

        self.opts = opts

        # create networks
        self.model_names=['deblur', 'flow']
        self.net_deblur=Deblur_net(opts.n_channels, opts.n_init_feat, opts.n_channels).cuda()
        self.net_flow=PWCDCNet().cuda()

        # print network
        self.print_networks(self.net_deblur)
        self.print_networks(self.net_flow)

        # load in initialized network parameters
        self.load_checkpoint(opts.model_label)

        self.upsampleX4 = nn.Upsample(scale_factor=4, mode='bilinear')

        if opts.is_training: 
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam([
                {'params': self.net_deblur.parameters()},
                {'params': self.net_flow.parameters()}], lr=opts.lr)
            self.optimizer_names = ['G']
            self.build_lr_scheduler()

            # create blurrer
            self.reblur_fn_prev=FlowBlurrer.create_with_implicit_mesh(opts.batch_sz,
                                                                    opts.n_channels,
                                                                    opts.crop_sz_H,
                                                                    opts.crop_sz_W,
                                                                    opts.max_ker_sz)

            self.reblur_fn_curr=FlowBlurrer.create_with_implicit_mesh(opts.batch_sz,
                                                                    opts.n_channels,
                                                                    opts.crop_sz_H,
                                                                    opts.crop_sz_W,
                                                                    opts.max_ker_sz)

            self.mask_fn=FlowWarpMask.create_with_implicit_mesh(opts.batch_sz,
                                                                opts.n_channels,
                                                                opts.crop_sz_H,
                                                                opts.crop_sz_W)

            # create loss functions
            self.loss_fn_data=L1Loss()
            self.loss_fn_tv_2C=VariationLoss(nc=2)
            self.loss_fn_tv_3C=VariationLoss(nc=3)

    def set_input(self, _input):
        im_blur_prev, im_blur_curr, mu_prev, mu_curr, dt, im_sharp_prev, im_sharp_curr=_input

        self.im_blur_prev=im_blur_prev.cuda()
        self.im_blur_curr=im_blur_curr.cuda()
        self.mu_prev=mu_prev.cuda().float()
        self.mu_curr=mu_curr.cuda().float()
        self.dt=dt.cuda().float()

        # used to monitor training progress
        if im_sharp_prev is not None:
            self.im_sharp_prev=im_sharp_prev.cuda()
            self.im_sharp_curr=im_sharp_curr.cuda()

    def forward(self):
        im_pred_prev=self.net_deblur(self.im_blur_prev)
        im_pred_curr=self.net_deblur(self.im_blur_curr)

        flows_prev_to_curr=self.net_flow(im_pred_prev.detach(), im_pred_curr.detach())
        flows_curr_to_prev=self.net_flow(im_pred_curr.detach(), im_pred_prev.detach())

        flow_prev_to_curr = self.upsampleX4(flows_prev_to_curr[0])*20.0
        flow_curr_to_prev = self.upsampleX4(flows_curr_to_prev[0])*20.0 

        return im_pred_prev, im_pred_curr, flow_prev_to_curr, flow_curr_to_prev

    def optimize_parameters(self):
        self.loss_tv_flow = torch.zeros([1], requires_grad=True).cuda()
        self.loss_tv_image = torch.zeros([1], requires_grad=True).cuda()
        
        #
        self.im_pred_prev, \
        self.im_pred_curr, \
        self.flow_prev_to_curr, \
        self.flow_curr_to_prev=self.forward()

        ##########################################################################
        #      Use bi-diretional optical flow to compute flow occlusion mask     #
        ##########################################################################
        self.mask_flow_prev = self.mask_fn(self.flow_curr_to_prev.detach())
        self.mask_flow_curr = self.mask_fn(self.flow_prev_to_curr.detach())

        ##########################################################################
        #             Compute LR & smooth losses to train FlowNet                #
        ##########################################################################
        self.syn_pred_prev_flowNet, _ = warp_image_flow(self.im_pred_curr.detach(), self.flow_prev_to_curr)
        self.syn_pred_curr_flowNet, _ = warp_image_flow(self.im_pred_prev.detach(), self.flow_curr_to_prev)

        self.syn_pred_prev_deblurNet, _ = warp_image_flow(self.im_pred_curr, self.flow_prev_to_curr.detach())
        self.syn_pred_curr_deblurNet, _ = warp_image_flow(self.im_pred_prev, self.flow_curr_to_prev.detach())

        # loss
        self.loss_lr = self.opts.lambda_lr *\
                            (self.loss_fn_data(self.syn_pred_prev_flowNet, self.im_pred_prev.detach(), self.mask_flow_prev.detach()) +\
                            self.loss_fn_data(self.syn_pred_curr_flowNet, self.im_pred_curr.detach(), self.mask_flow_curr.detach())) +\
                        self.opts.lambda_lr * 0.1 *\
                            (self.loss_fn_data(self.syn_pred_prev_deblurNet, self.im_pred_prev, self.mask_flow_prev.detach())+\
                            self.loss_fn_data(self.syn_pred_curr_deblurNet, self.im_pred_curr, self.mask_flow_curr.detach()))

        if self.opts.lambda_flow_tv>1e-6:
            self.loss_tv_flow += self.opts.lambda_flow_tv * \
                                (self.loss_fn_tv_2C(self.flow_prev_to_curr) + self.loss_fn_tv_2C(self.flow_curr_to_prev))

        ##########################################################################
        #             Compute self-consistency to train DeblurNet                #
        ##########################################################################        
        dc_prev = (0.5 * self.mu_prev / self.dt).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dc_curr = (0.5 * self.mu_curr / self.dt).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        half_flow_prev_to_curr = self.flow_prev_to_curr * dc_prev
        half_flow_curr_to_prev = self.flow_curr_to_prev * dc_prev

        self.syn_im_prev, self.mask_reblur_prev = self.reblur_fn_prev(self.im_pred_prev, half_flow_prev_to_curr)
        self.syn_im_curr, self.mask_reblur_curr = self.reblur_fn_curr(self.im_pred_curr, half_flow_curr_to_prev)

        # loss
        self.loss_self = self.loss_fn_data(self.syn_im_prev, self.im_blur_prev, self.mask_reblur_prev.detach()) +\
                        self.loss_fn_data(self.syn_im_curr, self.im_blur_curr, self.mask_reblur_curr.detach())

        if self.opts.lambda_img_tv:
            self.loss_tv_image = self.opts.lambda_img_tv*\
                                (self.loss_fn_tv_3C(self.im_pred_prev) + self.loss_fn_tv_3C(self.im_pred_curr))

        ##########################################################################
        #                            Optimize the network                        #
        ##########################################################################  
        self.loss_G = self.loss_lr + self.loss_self + self.loss_tv_flow + self.loss_tv_image        

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step() 

    def save_checkpoint(self, label):
        self.save_network(self.net_deblur, 'deblur', label, self.opts.log_dir)
        self.save_network(self.net_flow, 'flow', label, self.opts.log_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_deblur, 'deblur', label, self.opts.log_dir)
        self.load_network(self.net_flow, 'flow', label, self.opts.log_dir)

    def get_current_scalars(self):
        scalars = {}
        scalars['loss_G'] = self.loss_G.item()
        scalars['loss_lr'] = self.loss_lr.item()
        scalars['loss_self'] = self.loss_self.item()
        scalars['loss_tv_flow'] = self.loss_tv_flow.item()
        scalars['loss_tv_image'] = self.loss_tv_image.item()
        
        optimizer = getattr(self, 'optimizer_G')
        for param_group in optimizer.param_groups:
            scalars['lr'] = param_group['lr']
        
        return scalars

    def get_current_visuals(self):
        visuals = {}

        visuals['im_blur_curr'] = self.im_blur_curr
        visuals['im_pred_curr'] = self.im_pred_curr.clone().clamp(0.,1.)
        visuals['flow_curr_to_prev'] = torch.from_numpy(flow_to_numpy_rgb(self.flow_curr_to_prev).transpose(0,3,1,2)).float()/255.
        visuals['mask_lr_curr']=self.mask_flow_curr.clone().repeat(1,3,1,1)
        visuals['mask_reblur_curr']=self.mask_reblur_curr.clone().repeat(1,3,1,1)

        return visuals
