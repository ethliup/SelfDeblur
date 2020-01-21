import os
import sys
import torch
import argparse
import numpy as np
from tensorboardX import SummaryWriter

from dataloader import *
from model_self_deblur_net import *
from generic_train_test import *

##===================================================##
##********** Configure training settings ************##
##===================================================##
parser=argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=2, help='batch size used for training')
parser.add_argument('--continue_train', type=bool, default=False, help='flags used to indicate if train model from previous trained weight')
parser.add_argument('--crop_sz_H', type=int, default=256, help='cropped image size height')
parser.add_argument('--crop_sz_W', type=int, default=256, help='cropped image size width')
parser.add_argument('--is_training', type=bool, default=True, help='flag used for selecting training mode or evaluation mode')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of optimizer')
parser.add_argument('--lr_step', type=int, default=40, help='lr decay rate')
parser.add_argument('--lr_start_epoch_decay', type=int, default=260, help='epoch to start lr decay')
parser.add_argument('--max_ker_sz', type=int, default=30, help='maximum blur kernel size in pixels')
parser.add_argument('--n_channels', type=int, default=3, help='number of channels of input/output image')
parser.add_argument('--n_init_feat', type=int, default=32, help='number of channels of initial features')
parser.add_argument('--seq_len', type=int, default=2)
parser.add_argument('--shuffle_data', type=bool, default=True)

parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--max_epochs', type=int, default=501)
parser.add_argument('--save_freq', type=int, default=20)
parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--model_label', type=str, default='pretrained', help='label used to load pre-trained model')

parser.add_argument('--lambda_lr', type=float, default=2.0)
parser.add_argument('--lambda_img_tv', type=float, default=0.)
parser.add_argument('--lambda_flow_tv', type=float, default=0.)

parser.add_argument('--dataset_root_dir', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)

opts=parser.parse_args() 

##===================================================##
##*************** Create dataloader *****************##
##===================================================##
dataloader = Create_dataloader(opts)

##===================================================##
##*************** Create datalogger *****************##
##===================================================##
logger = SummaryWriter(opts.log_dir)

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelSelfDeblurNet(opts)

##===================================================##
##**************** Train the network ****************##
##===================================================##
class Train(Generic_train_test):
    def decode_input(self, data):
        im_blur = data['A']
        im_sharp = data['B']
        mu = data['C']
        timestamps = data['D']

        im_blur_prev = im_blur[:,:self.opts.n_channels,:,:].clone()
        im_blur_curr = im_blur[:,self.opts.n_channels:self.opts.n_channels*2,:,:].clone()

        im_sharp_prev = im_sharp[:,:self.opts.n_channels,:,:].clone()
        im_sharp_curr = im_sharp[:,self.opts.n_channels:self.opts.n_channels*2,:,:].clone()
                
        t_prev = timestamps[:,0].clone()
        t_curr = timestamps[:,1].clone()
        dt = torch.abs(t_curr - t_prev)

        mu_prev = mu[:,0].clone()
        mu_curr = mu[:,1].clone()

        return [im_blur_prev, im_blur_curr, mu_prev, mu_curr, dt, im_sharp_prev, im_sharp_curr]

Train(model, opts, dataloader, logger).train()

