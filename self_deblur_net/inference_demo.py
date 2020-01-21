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
parser=argparse.ArgumentParser()
parser.add_argument('--continue_train', type=bool, default=False, help='flags used to indicate if train model from previous trained weight')
parser.add_argument('--is_training', type=bool, default=False, help='flag used for selecting training mode or evaluation mode')
parser.add_argument('--n_channels', type=int, default=3, help='number of channels of input/output image')
parser.add_argument('--n_init_feat', type=int, default=32, help='number of channels of initial features')

parser.add_argument('--compute_metrics', action='store_true')
parser.add_argument('--save_images', action='store_true')

parser.add_argument('--demo_data_dir', type=str, required=True)
parser.add_argument('--model_label', type=str, required=True, help='label used to load pre-trained model')
parser.add_argument('--log_dir', type=str, required=True, help='directory used to store pre-trained models')
parser.add_argument('--results_dir', type=str, required=True, help='directory used to store experimental results')

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
		all_files = os.listdir(opts.demo_data_dir)

		for i in range(len(all_files)):
			f = all_files[i]
			if not is_image_file(f):
				continue

			im_blur = os.path.join(opts.demo_data_dir, f)
			im_blur = io.imread(im_blur, plugin='pil') 
			if im_blur.ndim==2:
				im_blur=np.expand_dims(im_blur, axis=2)
				im_blur=np.repeat(im_blur, 3, axis=2)
			im_blur=im_blur[:,:,:3]
			im_blur = torch.from_numpy(im_blur.transpose(2,0,1)).float().cuda()/255.
			im_blur = im_blur[:3,:,:].clone().unsqueeze(0)
			_input = [im_blur, None]
			self.test_individual(_input, opts.results_dir, i)

Test(model, opts, None, None).demo()

