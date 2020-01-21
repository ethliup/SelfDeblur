#!/bin/bash

# !! Updata the path to the datasets with your own local path !!
root_path_fastec_test_data=/home/peidong/Desktop/Fastec_dataset/test
root_path_real_test_data=/home/peidong/Desktop/Real_dataset/test

# create an empty folder for experimental results
mkdir -p experiments/results_full_fastec
mkdir -p experiments/results_full_real

# invoke python scripts for deblurring
cd self_deblur_net

python inference.py \
            --save_images \
            --compute_metrics \
            --dataset_root_dir=$root_path_fastec_test_data \
            --model_label='Fastec_pretrained' \
            --log_dir=../experiments/pretrained_models \
            --results_dir=../experiments/results_full_fastec

python inference.py \
            --save_images \
            --dataset_root_dir=$root_path_real_test_data \
            --model_label='Real_pretrained' \
            --log_dir=../experiments/pretrained_models \
            --results_dir=../experiments/results_full_real
