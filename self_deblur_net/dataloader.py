import os
import torch
import random
import numpy as np
from skimage import io
from torch.utils.data import Dataset

from transforms import *

class Dataset_reader(Dataset):
    def __init__(self, root_dir, transform=None, nseq=1, nmax_per_seq=1000000):
        self.transform=transform
        self.nseq=nseq

        self.blur_image_paths=[]
        self.sharp_image_paths=[]
        self.exposure=[]
        self.timestamp=[]

        f_paired_image_list=open(root_dir+'/paired_image_list.log','r')
        
        list_blur = []
        list_sharp = []
        list_exposure = []
        list_timestamp = []
        ncounter=0

        for line in f_paired_image_list:
            if '#' in line:
                continue

            if '*' in line:
                list_blur=[]
                list_sharp=[]
                list_exposure = []
                list_timestamp = []
                ncounter=0
                continue

            line = line.replace('\n', '')
            line_ls = line.split(' ')
            blur_path=root_dir+'/'+line_ls[0]
            sharp_path=root_dir+'/'+line_ls[1]

            list_blur.append(blur_path)
            list_sharp.append(sharp_path)
            list_exposure.append(float(line_ls[2]))
            list_timestamp.append(float(line_ls[3]))
                    
            if len(list_blur)<nseq:
                continue

            ncounter+=1
            if ncounter<nmax_per_seq:
                self.blur_image_paths.append(list_blur.copy())
                self.sharp_image_paths.append(list_sharp.copy())
                self.exposure.append(list_exposure.copy())
                self.timestamp.append(list_timestamp.copy())

            list_blur.pop(0)
            list_sharp.pop(0)   
            list_exposure.pop(0)
            list_timestamp.pop(0)

    def __getitem__(self, idx):
        list_blur_path=self.blur_image_paths[idx]
        list_sharp_path=self.sharp_image_paths[idx]

        blur_images=io.imread(list_blur_path[0], plugin='pil')
        sharp_images=io.imread(list_sharp_path[0], plugin='pil')

        if blur_images.ndim==2:
            blur_images=np.expand_dims(blur_images, axis=2)
            sharp_images=np.expand_dims(sharp_images, axis=2)
            blur_images=np.repeat(blur_images, 3, axis=2)
            sharp_images=np.repeat(sharp_images, 3, axis=2)

        blur_images=blur_images[:,:,:3]
        sharp_images=sharp_images[:,:,:3]

        for i in range(1, self.nseq):
            blur_image=io.imread(list_blur_path[i], plugin='pil')
            sharp_image=io.imread(list_sharp_path[i], plugin='pil')

            if blur_image.ndim==2:
                blur_image=np.expand_dims(blur_image, axis=2)
                sharp_image=np.expand_dims(sharp_image, axis=2)
                blur_image=np.repeat(blur_image, 3, axis=2)
                sharp_image=np.repeat(sharp_image, 3, axis=2)

            blur_image=blur_image[:,:,:3]
            sharp_image=sharp_image[:,:,:3]

            blur_images=np.concatenate((blur_images, blur_image),axis=2)
            sharp_images=np.concatenate((sharp_images, sharp_image), axis=2)

        blur_images=blur_images.astype(np.float)
        sharp_images=sharp_images.astype(np.float)

        if self.transform:
            blur_images, sharp_images=self.transform([blur_images, sharp_images])

        exposures = torch.FloatTensor(self.exposure[idx])
        timestamp = torch.FloatTensor(self.timestamp[idx])

        return {'A': blur_images,
                'B': sharp_images,
                'C': exposures,
                'D': timestamp}

    def __len__(self):
        return len(self.blur_image_paths)

def Create_dataloader(opts):
    data_augmentor=Compose([Merge(),
                            Normalize(),
                            Random_crop(size=[opts.crop_sz_H, opts.crop_sz_W]),
                            Split([0, opts.n_channels*opts.seq_len], [opts.n_channels*opts.seq_len, 2*opts.n_channels*opts.seq_len]),
                            [To_tensor(), To_tensor()],])

    dataset_reader=Dataset_reader(opts.dataset_root_dir, data_augmentor, opts.seq_len)

    dataloader=torch.utils.data.DataLoader(dataset_reader,
                                        batch_size=opts.batch_sz, 
                                        shuffle=opts.shuffle_data, 
                                        num_workers=opts.batch_sz, 
                                        drop_last=True)
    return dataloader

def Create_dataloader_inference(opts):
    data_augmentor=Compose([Merge(),
                            Normalize(),
                            Crop(size=[opts.crop_sz_H, opts.crop_sz_W]),
                            Split([0, opts.n_channels*opts.seq_len], [opts.n_channels*opts.seq_len, 2*opts.n_channels*opts.seq_len]),
                            [To_tensor(), To_tensor()],])

    dataset_reader=Dataset_reader(opts.dataset_root_dir, data_augmentor, opts.seq_len)

    dataloader=torch.utils.data.DataLoader(dataset_reader,
                                        batch_size=opts.batch_sz, 
                                        shuffle=opts.shuffle_data, 
                                        num_workers=opts.batch_sz, 
                                        drop_last=True)
    return dataloader

