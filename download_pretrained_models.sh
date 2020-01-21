#!/bin/bash

# create an empty folder for pretrained models 
mkdir -p experiments/pretrained_models

# download pretrained models 
cd experiments/pretrained_models
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VlnQrGMt3gDU4ocTLhIDyndccj30CA1M' -O Fastec_pretrained_net_deblur.pth
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1y-z46c4wpD_Tm5Vr6_aeYwK8GPUN6H-C' -O Real_pretrained_net_deblur.pth
