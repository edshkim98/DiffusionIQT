from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util

import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import math
import glob
import os
import random
from torch.utils.data.dataloader import default_collate

def my_collate(batch):
    batch = list(filter(lambda x : x is not None, batch))
    return default_collate(batch)

class supervisedIQT(Dataset):
    def __init__(self, lr_files, hr_files, train=True):
        self.lr_files = lr_files
        self.hr_files = hr_files
        self.patch_size = 32
        self.ratio = 0.8
        self.total_voxel = self.patch_size * self.patch_size * self.patch_size
        self.train = train
        self.is_transform = True

        self.mean_lr = 35.493949511348724
        self.std_lr = 37.11344433531084
        self.files_lr = []
        self.files_hr = []
        
        for i in range(len(self.lr_files)):
            self.files_lr.append(self.lr_files[i])
            self.files_hr.append(self.hr_files[i])

    def __len__(self):
        return len(self.files_lr)

    def transform(self, image): # transform 3D array to tensor
        
        image_torch = torch.FloatTensor(image)
        image_torch = (image_torch - self.mean_lr)/self.std_lr
        
        return image_torch
    
    def cube(self,data):

        hyp_norm = data

        if len(hyp_norm.shape)>3:
            hyp_norm = hyp_norm[:,:, 2:258, 27:283]
        else:
            hyp_norm = hyp_norm[2:258, 27:283]

        return hyp_norm

    def __getitem__(self, idx):
        
        self.lr = self.files_lr[idx]
        self.hr = self.files_hr[idx] 
        
        self.lr = nib.load(self.lr)
        self.lr_affine = self.lr.affine
        self.lr = torch.tensor(self.lr.get_fdata().astype(np.float32))
        self.img_shape = self.lr.shape
        self.hr = nib.load(self.hr)
        self.hr_affine = self.hr.affine
        self.hr = torch.tensor(self.hr.get_fdata().astype(np.float32))
        
        #Cube 
        self.lr = self.cube(self.lr)
        self.hr = self.cube(self.hr)
            
        random_idx = np.random.randint(low=0, high=256-self.patch_size+1, size=3)
        self.lr = self.lr[random_idx[0]:random_idx[0]+self.patch_size, random_idx[1]:random_idx[1]+self.patch_size, random_idx[2]:random_idx[2]+self.patch_size]
        self.hr = self.hr[random_idx[0]:random_idx[0]+self.patch_size, random_idx[1]:random_idx[1]+self.patch_size, random_idx[2]:random_idx[2]+self.patch_size]

        non_zero = np.count_nonzero(self.lr)
        non_zero_proportion = (non_zero/self.total_voxel)
        if (non_zero_proportion < self.ratio):
            return self.__getitem__(idx)
                        
        self.lr = self.transform(self.lr)
        self.hr = self.transform(self.hr)
            
        sample_lr = torch.unsqueeze(self.lr, 0)
        sample_hr = torch.unsqueeze(self.hr, 0)
        
        if self.train:
            #fname = self.files_lr[idx].split('/')[6]
            return sample_hr, sample_lr#, 'Index': idx, 'Affine': self.hr_affine}
        else:
            #fname = self.files_lr[idx].split('/')[7]
            return sample_hr, sample_lr#, 'Index': idx, 'Affine': self.hr_affine}
