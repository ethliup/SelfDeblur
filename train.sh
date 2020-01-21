#!/bin/bash

# !! Updata the path to the dataset and directory to 
# !! save your trained models with your own local path !!
root_path_training_data=/path
log_dir_deblur_net=/path
log_dir_flow_net=/path
log_dir_self_deblur_net=/path

#
cd self_deblur_net

# train the deblur net with 'blur_img -> blur_img' for identity mapping
python train_deblur_net.py \
          --dataset_root_dir=$root_path_training_data \
          --log_dir=$log_dir_deblur_net \
          --blur2blur

# train the flow net with blurry image in a self-supervised manner
python train_self_flow_net.py \
          --dataset_root_dir=$root_path_training_data \
          --log_dir=$log_dir_flow_net \
          --blur_input

# train the self-deblur net 
cp ${log_dir_deblur_net}'/30_net_deblur.pth' ${log_dir_self_deblur_net}'/pretrained_net_deblur.pth' 
cp ${log_dir_flow_net}'/200_net_flow.pth' ${log_dir_self_deblur_net}'/pretrained_net_flow.pth' 

python train_self_deblur_net.py \
          --dataset_root_dir=$root_path_training_data \
          --log_dir=$log_dir_self_deblur_net

