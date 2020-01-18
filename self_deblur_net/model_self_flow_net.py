import torch
import torch.nn as nn

from losses import *
from net_pwc import *
from model_base import *
from reblur_package import *
from flow_utils import *
 
class ModelSelfFlowNet(ModelBase):
    def __init__(self, opts):
        super(ModelSelfFlowNet, self).__init__()
        self.opts = opts
        
        # create network
        self.model_names = ['flow']
        self.net_flow = PWCDCNet().cuda()
 
        # print network
        self.print_networks(self.net_flow)

        if not opts.is_training or opts.continue_train:
            self.load_checkpoint(opts.model_label)

        self.upsampleX4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.downsampleX2 = nn.AvgPool2d(2, stride=2)
        
        # initialize mesh faces
        if opts.is_training:
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net_flow.parameters(), lr=opts.lr)
            self.optimizer_names = ['G']
            self.build_lr_scheduler()
            
            # define loss functions
            self.loss_fn_data = L1Loss()

            # create flow blurrer
            self.mask_fn = [None] * self.opts.pwc_lvls
            for l in range(self.opts.pwc_lvls):
                scale = 2 ** -(l+2) # we start at quarter resolution
                H = int(scale * opts.crop_sz_H)
                W = int(scale * opts.crop_sz_W)
                self.mask_fn[l] = FlowWarpMask.create_with_implicit_mesh(opts.batch_sz,2,H,W)

    def set_input(self, _input):
        imgs, = _input
        im_prev = imgs[:,:self.opts.n_channels,:,:]
        im_curr = imgs[:,self.opts.n_channels:self.opts.n_channels*2,:,:]
        self.im_prev = im_prev.cuda()
        self.im_curr = im_curr.cuda()

    def forward(self):
        pred_flows_prev_to_curr = self.net_flow(self.im_prev, self.im_curr)
        pred_flows_curr_to_prev = self.net_flow(self.im_curr, self.im_prev)

        pred_flow_prev_to_curr_0 = self.upsampleX4(pred_flows_prev_to_curr[0])*20.0
        pred_flow_curr_to_prev_0 = self.upsampleX4(pred_flows_curr_to_prev[0])*20.0 
        pred_flow_prev_to_curr_1 = self.upsampleX4(pred_flows_prev_to_curr[1])*10.0  # 20 / (2 ^ 1)
        pred_flow_curr_to_prev_1 = self.upsampleX4(pred_flows_curr_to_prev[1])*10.0  # 20 / (2 ^ 1)
        pred_flow_prev_to_curr_2 = self.upsampleX4(pred_flows_prev_to_curr[2])*5.0   # 20 / (2 ^ 2)
        pred_flow_curr_to_prev_2 = self.upsampleX4(pred_flows_curr_to_prev[2])*5.0   # 20 / (2 ^ 2)
        
        return pred_flows_prev_to_curr, \
               pred_flows_curr_to_prev, \
               [pred_flow_prev_to_curr_0, pred_flow_prev_to_curr_1, pred_flow_prev_to_curr_2], \
               [pred_flow_curr_to_prev_0, pred_flow_curr_to_prev_1, pred_flow_curr_to_prev_2]

    def optimize_parameters(self):
        pred_flows_prev_to_curr, \
        pred_flows_curr_to_prev, \
        self.pred_flow_prev_to_curr_0, \
        self.pred_flow_curr_to_prev_0 = self.forward()

        im_prev_clone = self.im_prev.clone()
        im_curr_clone = self.im_curr.clone()

        res = (im_prev_clone.shape[2] * im_prev_clone.shape[3]) / self.opts.pwc_lvls
        pyr_weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        pyr_weights = [i * res for i in pyr_weights]
        # import pdb; pdb.set_trace()

        self.syn_im_prev = [None] * self.opts.pwc_lvls
        self.syn_im_curr = [None] * self.opts.pwc_lvls
        self.syn_flow_prev_to_curr = [None] * self.opts.pwc_lvls
        self.syn_flow_curr_to_prev = [None] * self.opts.pwc_lvls
        self.syn_flow_prev_to_curr = [None] * self.opts.pwc_lvls
        self.syn_flow_curr_to_prev = [None] * self.opts.pwc_lvls
        self.mask_im_prev = [None] * self.opts.pwc_lvls
        self.mask_im_curr = [None] * self.opts.pwc_lvls

        self.loss_lr_image = torch.zeros([1], requires_grad=True).cuda()
        self.loss_lr_flow = torch.zeros([1], requires_grad=True).cuda()

        for l in range(self.opts.pwc_lvls):
            if l > 0:
                im_prev_clone = self.downsampleX2(im_prev_clone)
                im_curr_clone = self.downsampleX2(im_curr_clone)
                
            fs = 20.0 / (2 ** l)
            self.mask_im_prev[l] = self.upsampleX4(self.mask_fn[l](pred_flows_prev_to_curr[l].detach()*fs).float())
            self.mask_im_curr[l] = self.upsampleX4(self.mask_fn[l](pred_flows_curr_to_prev[l].detach()*fs).float())

            self.syn_im_prev[l], _ = warp_image_flow(im_curr_clone, self.pred_flow_prev_to_curr_0[l])
            self.syn_im_curr[l], _ = warp_image_flow(im_prev_clone, self.pred_flow_curr_to_prev_0[l])

            self.syn_flow_prev_to_curr[l], _ = warp_image_flow(self.pred_flow_curr_to_prev_0[l], self.pred_flow_prev_to_curr_0[l])
            self.syn_flow_curr_to_prev[l], _ = warp_image_flow(self.pred_flow_prev_to_curr_0[l], self.pred_flow_curr_to_prev_0[l])
            self.syn_flow_prev_to_curr[l] = -1.*self.syn_flow_prev_to_curr[l]
            self.syn_flow_curr_to_prev[l] = -1.*self.syn_flow_curr_to_prev[l]

            self.loss_lr_image += pyr_weights[l] * (self.loss_fn_data(self.syn_im_prev[l], im_prev_clone, self.mask_im_prev[l], True) \
                                                    + self.loss_fn_data(self.syn_im_curr[l], im_curr_clone, self.mask_im_curr[l], True))

            self.loss_lr_flow += pyr_weights[l] * self.opts.pwc_fwdbwd * \
                                (self.loss_fn_data(self.syn_flow_prev_to_curr[l], self.pred_flow_prev_to_curr_0[l], self.mask_im_prev[l], True) +\
                                 self.loss_fn_data(self.syn_flow_curr_to_prev[l], self.pred_flow_curr_to_prev_0[l], self.mask_im_curr[l], True))

        self.loss_G = self.loss_lr_image + self.loss_lr_flow

        #=============== Optimize generator =============#
        self.optimizer_G.zero_grad() 
        self.loss_G.backward()
        self.optimizer_G.step()  

    # save networks to file 
    def save_checkpoint(self, label):
        self.save_network(self.net_flow, 'flow', label, self.opts.log_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_flow, 'flow', label, self.opts.log_dir)

    def get_current_scalars(self):
        losses = {}
        losses['loss_lr_image'] = self.loss_lr_image.item()
        losses['loss_lr_flow'] = self.loss_lr_flow.item()
        losses['loss_G'] = self.loss_G.item()
        return losses

    def get_current_visuals(self):
        output_visuals = {}
        
        output_visuals['im_prev'] = self.im_prev
        output_visuals['im_curr'] = self.im_curr

        output_visuals['pred_flows_prev_to_curr_0'] = torch.from_numpy(flow_to_numpy_rgb(self.pred_flow_prev_to_curr_0[0]).transpose(0,3,1,2)).float()/255.
        output_visuals['pred_flows_prev_to_curr_1'] = torch.from_numpy(flow_to_numpy_rgb(self.pred_flow_prev_to_curr_0[1]).transpose(0,3,1,2)).float()/255.
        output_visuals['pred_flows_prev_to_curr_2'] = torch.from_numpy(flow_to_numpy_rgb(self.pred_flow_prev_to_curr_0[2]).transpose(0,3,1,2)).float()/255.

        output_visuals['mask_im_curr']=self.mask_im_curr[0].clone().repeat(1,3,1,1)

        return output_visuals
