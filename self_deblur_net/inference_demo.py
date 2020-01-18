import os
import sys
import torch
import argparse
import numpy as np
from skimage import io

from model_deblur_net import *
from generic_train_test import *

##===================================================##
##********** Configure training settings ************##
##===================================================##
path_root_dir = '/home/peidong/leonhard/project/infk/cvg/liup/'
demo_data_dir = '/home/peidong/Desktop/demo'

parser=argparse.ArgumentParser()
parser.add_argument('--continue_train', type=bool, default=False, help='flags used to indicate if train model from previous trained weight')
parser.add_argument('--is_training', type=bool, default=False, help='flag used for selecting training mode or evaluation mode')
parser.add_argument('--n_channels', type=int, default=3, help='number of channels of input/output image')
parser.add_argument('--n_init_feat', type=int, default=32, help='number of channels of initial features')

parser.add_argument('--compute_metrics', action='store_true')
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--model_label', type=str, default='320', help='label used to load pre-trained model')

parser.add_argument('--log_dir', type=str, default=path_root_dir+'/logs/deblur/net_self_deblur/Fastec_reblur_conv_lambda_im_tv_0.0_lambda_flow_tv_0.0', help='directory used to store trained networks')

opts=parser.parse_args()

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelDeblurNet(opts)

##===================================================##
##**************** Test the network ****************##
##===================================================##
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.tif',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class Test(Generic_train_test):
	def demo(self):
		all_files = os.listdir(demo_data_dir)
		dir_results = os.path.join(demo_data_dir, 'results')
		if not os.path.exists(dir_results):
			os.makedirs(dir_results)

		for i in range(len(all_files)):
			f = all_files[i]
			if not is_image_file(f):
				continue

			im_blur = os.path.join(demo_data_dir, f)
			im_blur = io.imread(im_blur)
			im_blur = torch.from_numpy(im_blur.transpose(2,0,1)).float().cuda()/255.
			im_blur = im_blur[:3,:,:].clone().unsqueeze(0)
			_input = [im_blur, None]
			self.test_individual(_input, dir_results, i)

Test(model, opts, None, None).demo()

