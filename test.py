import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import glob
import time
import yaml
from yaml.loader import SafeLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import cv2
from imagen_pytorch3D import Unet, NullUnet, Imagen, SRUnet256, alpha_cosine_log_snr

from trainer import ImagenTrainer
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image 
import PIL 
import pandas as pd
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce
from einops_exts import rearrange_many, repeat_many, check_shape
from torch import nn, einsum

from data import supervisedIQT_INF, my_collate
from utils_mine import *

import time
from metrics import * 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cube(data):

    hyp_norm = data

    if len(hyp_norm.shape)>3:
        hyp_norm = hyp_norm[:,:, 2:258, 27:283]
    else:
        hyp_norm = hyp_norm[2:258, 27:283]

    return hyp_norm

with open('./config/eval_config.yaml','r') as file:
    configs = yaml.safe_load(file)

lrfile =  '/cluster/project0/IQT_Nigeria/skim/HCP_Harry_x8/test_small/800941/T1w/lr_norm.nii.gz'
hrfile = '/cluster/project0/IQT_Nigeria/skim/HCP_Harry_x8/test_small/800941/T1w/T1w_acpc_dc_restore_brain.nii.gz'

if configs['Train']['batch_sample']:
    batch_size = 1
else:
    batch_size = configs['Eval']['batch_size']

dataset = supervisedIQT_INF(configs, lrfile)
test_loader =  DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate, shuffle=False, drop_last=False)

batch_size = configs['Eval']['batch_size']

min_bound = (0. - configs['Data']['mean'])/ configs['Data']['std']

# unet for imagen
unet1 = NullUnet()
unet2 = SRUnet256(
        dim = 64,
        dim_mults = (1, 2, 4),
        channels=1,
        num_resnet_blocks = (2, 2, 2), #2,4,4
        init_conv_kernel_size = 3,
        lowres_cond = True,
        init_cross_embed = False,
        init_cross_embed_kernel_sizes = (3, 5, 7),
        att_type = configs['Train']['att_type'],
        attn_dim_head = configs['Train']['att_head_dim'],
        attend_at_middle = configs['Train']['att_mid'],
        attend_at_middle_depth = configs['Train']['att_mid_depth'],
        attend_at_middle_heads = configs['Train']['att_mid_heads'],
        attend_at_enc = configs['Train']['att_enc'],
        attend_at_enc_depth = configs['Train']['att_enc_depth'],
        attend_at_enc_heads = configs['Train']['att_enc_heads'],
        att_drop = configs['Train']['att_drop'],
        att_forward_drop = configs['Train']['att_forward_drop'],
        att_forward_expansion = configs['Train']['att_forward_expansion'],
        att_skip_scale = configs['Train']['skip_scale'],
        att_localvit = configs['Train']['att_localvit'],
        groups = configs['Train']['num_groups'],
        emb_size = configs['Train']['emb_size'],
        init_dim = 64,
        memory_efficient = configs['Train']['efficient'],
        use_se_attn = True,
        pixel_shuffle_upsample = True,
        boundary = configs['Train']['boundary'],
        batch_sample = configs['Train']['batch_sample'],

)
print("Unet2 loaded")
imagen = Imagen(
        configs = configs,
        unets = (unet1, unet2),
        min_bound = min_bound,
        image_sizes = (configs['Train']['patch_size_sub'], configs['Train']['patch_size_sub']),#(32, 32),
        channels=1,
        pred_objectives = configs['Train']['pred_obj'],
        timesteps = configs['Train']['timesteps'],
        dynamic_thresholding = configs['Train']['dynamic_threshold'],
        p2_loss_weight_gamma = 0.0,
        auto_normalize_img = False,
        cond_drop_prob = 0.0,
        lpips = configs['Train']['lpips'],
        boundary = configs['Train']['boundary']

).to(device)
print("Imagen loaded")
trainer = ImagenTrainer(
    configs = configs,
    imagen = imagen,
    gradient_accumulation_steps = 4,
    #cosine_decay_max_steps = len(train_loader)*50,
    split_valid_from_train = False # whether to split the validation dataset from the training
)

trainer.load('/cluster/project0/IQT_Nigeria/skim/diffusion/results/Final_app_32_8x/model/3dimagen.pt')

params = sum([np.prod(p.size()) for p in unet2.parameters()])
print("Number of params: ", params)
print("Number of time steps: ", imagen.noise_schedulers[1].num_timesteps)

lowres = nib.load(lrfile)
highres = nib.load(hrfile)
affine = highres.affine

lowres = lowres.get_fdata()
highres = highres.get_fdata()

if lowres.shape[-1] != 256:
    lowres = cube(lowres)
    highres = cube(highres)
low, high = 8, 248
lowres = lowres[low:high, low:high, low:high]
highres = highres[low:high, low:high, low:high]

print(f'lwores: {lowres.shape} highres: {highres.shape}')

lowres_torch = torch.unsqueeze(torch.unsqueeze(torch.tensor(lowres),0),0)
highres_torch = torch.unsqueeze(torch.unsqueeze(torch.tensor(highres),0),0)

mean, std = configs['Data']['mean'], configs['Data']['std'] 
pred_ary = torch.zeros(lowres.shape)
pred_ary = (pred_ary - mean)/std
highres = (highres - mean)/std
lowres = (lowres - mean)/std
min_val = lowres.min()

patch_size = configs['Train']['patch_size_sub']
total_voxel = patch_size*patch_size*patch_size
op = configs['Eval']['overlap'] // 2
print("Start inferencing!")

start = time.time()

for i,data in enumerate(test_loader):
    if data is not None:
        patch_input, idx = data
        print(patch_input.shape, idx)
        if patch_input.shape[-1] != configs['Train']['patch_size_sub']:
            patch_input = convertVolume2subVolume(patch_input) #convert 96 to 32
            print("Converted: ", patch_input.shape) 
        patch_input = patch_input.to(device)
        outputs = trainer.sample(batch_size = patch_input.shape[0], skip_steps=None, return_all_unet_outputs = True, return_pil_images = False, start_image_or_video = patch_input, start_at_unet_number = 2)
        outputs = outputs[0].cpu()
        if not (configs['Train']['boundary'] or configs['Train']['batch_sample']):
            for j in range(patch_input.shape[0]):   
                if configs['Eval']['overlap'] < configs['Train']['patch_size_sub']:
                    if (0 in idx[0]) or (pred_ary.shape[-1] in idx[0] + patch_size): #if starting idx... 

                        op_x_start, op_x_end, op_y_start, op_y_end, op_z_start, op_z_end = op, op, op, op ,op ,op

                        if (idx[0][0] == 0) or (pred_ary.shape[-1] == idx[0][0] + patch_size):
                            if (idx[0][0] == 0):
                                op_x_start = 0
                            if (pred_ary.shape[-1] == idx[0][0] + patch_size):
                                op_x_end = 0
                            
                        if (idx[0][1] == 0) or (pred_ary.shape[-1] == idx[0][1] + patch_size):
                            if (idx[0][1] == 0):
                                op_y_start = 0
                            if (pred_ary.shape[-1] == idx[0][1] + patch_size):
                                op_y_end = 0
 
                        if (idx[0][2] == 0) or (pred_ary.shape[-1] == idx[0][2] + patch_size):
                            if (idx[0][2] == 0):
                                op_z_start = 0
                            if (pred_ary.shape[-1] == idx[0][2] + patch_size):
                                op_z_end = 0
                        pred_ary[idx[0][0]+op_x_start: idx[0][0]+ patch_size-op_x_end, idx[0][1]+op_y_start: idx[0][1]+patch_size-op_y_end, idx[0][2]+op_z_start: idx[0][2]+patch_size-op_z_end] = outputs[0][0][op_x_start:patch_size-op_x_end,op_y_start:patch_size-op_y_end,op_z_start:patch_size-op_z_end]

                    else:
                        pred_ary[idx[j][0]+op:idx[j][0]+patch_size-op,idx[j][1]+op:idx[j][1]+patch_size-op,idx[j][2]+op:idx[j][2]+patch_size-op] = outputs[j,0][op:patch_size-op,op:patch_size-op,op:patch_size-op]
                else:
                    pred_ary[idx[j][0]:idx[j][0]+patch_size,idx[j][1]:idx[j][1]+patch_size,idx[j][2]:idx[j][2]+patch_size] = outputs[j][0]
        else:
            if configs['Train']['batch_sample']: #(configs['Train']['boundary'] and configs['Train']['batch_sample']):
                outputs = merge_sub_volumes(outputs)
                
            patch_size = outputs.shape[-1]
            if configs['Eval']['overlap'] < patch_size:
                if (0 in idx[0]) or (pred_ary.shape[-1] in idx[0] + patch_size): #if starting idx...
                    print("Starting idx: ", idx) 
                    op_x_start, op_x_end, op_y_start, op_y_end, op_z_start, op_z_end = op, op, op, op ,op ,op
                    if (idx[0][0] == 0) or (pred_ary.shape[-1] == idx[0][0] + patch_size):
                        if (idx[0][0] == 0):
                            op_x_start = 0
                        if (pred_ary.shape[-1] == idx[0][0] + patch_size):
                            op_x_end = 0
                    if (idx[0][1] == 0) or (pred_ary.shape[-1] == idx[0][1] + patch_size):
                        if (idx[0][1] == 0):
                            op_y_start = 0
                        if (pred_ary.shape[-1] == idx[0][1] + patch_size):
                            op_y_end = 0
                    if (idx[0][2] == 0) or (pred_ary.shape[-1] == idx[0][2] + patch_size):
                        if (idx[0][2] == 0):
                            op_z_start = 0
                        if (pred_ary.shape[-1] == idx[0][2] + patch_size):
                            op_z_end = 0
                    pred_ary[idx[0][0]+op_x_start: idx[0][0]+ patch_size-op_x_end, idx[0][1]+op_y_start: idx[0][1]+patch_size-op_y_end, idx[0][2]+op_z_start: idx[0][2]+patch_size-op_z_end] = outputs[0][0][op_x_start:patch_size-op_x_end,op_y_start:patch_size-op_y_end,op_z_start:patch_size-op_z_end]
                else:
                    print("Int idx: ", idx)
                    pred_ary[idx[0][0]+op: idx[0][0]+ patch_size-op, idx[0][1]+op: idx[0][1]+patch_size-op, idx[0][2]+op: idx[0][2]+patch_size-op] = outputs[0][0][op:patch_size-op,op:patch_size-op,op:patch_size-op]
            else:
                pred_ary[idx[0][0]: idx[0][0]+ patch_size, idx[0][1]: idx[0][1]+patch_size, idx[0][2]: idx[0][2]+patch_size] = outputs[0][0]
#pred_ary[np.where(lowres==min_val)] = min_val
end = time.time()
print("TIME: {}".format(end-start))
np.save('volume_inf.npy', pred_ary.numpy())
np.save('volume_gt.npy', highres)
np.save('volume_lr.npy', lowres)

nib_img = nib.Nifti1Image(pred_ary.numpy(), affine)
nib.save(nib_img, 'volume_inf.nii.gz')
nib_img = nib.Nifti1Image(highres, affine)
nib.save(nib_img, 'volume_gt.nii.gz')
nib_img = nib.Nifti1Image(lowres, affine)
nib.save(nib_img, 'volume_lr.nii.gz')

