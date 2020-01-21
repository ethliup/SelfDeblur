#!/bin/bash

# create an empty folder for experimental results
mkdir -p experiments/results_demo_fastec
mkdir -p experiments/results_demo_real

# invoke python scripts for deblurring
cd self_deblur_net
# python inference_demo.py \
#             --save_images \
#             --demo_data_dir=../demo/fastec \
#             --model_label='Fastec_pretrained' \
#             --log_dir=../experiments/pretrained_models \
#             --results_dir=../experiments/results_demo_fastec

python inference_demo.py \
            --save_images \
            --demo_data_dir=../demo/real \
            --model_label='Real_pretrained' \
            --log_dir=../experiments/pretrained_models \
            --results_dir=../experiments/results_demo_real
