import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import yaml

#import cv2
from imagen_pytorch3D import Unet, NullUnet, Imagen, SRUnet256, alpha_cosine_log_snr
from data import IQTDataset, supervisedIQT
from trainer import ImagenTrainer

import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn.functional as F

from utils_mine import set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    
    set_seed(42)
    
    eval_step = 50

    with open('./config/config.yaml','r') as file:
        configs = yaml.safe_load(file)

    hr_files = glob.glob(configs['Data']['groundtruth_path'])
    lr_files = glob.glob(configs['Data']['lowres_path'])
    if configs['Train']['batch_sample']:
        batch_size = 1
    else:
        batch_size = configs['Train']['batch_size']
    project_path = configs['Results'] + configs['ProjectName']
    
    assert os.path.isdir(project_path) == False, f"project {project_path} exists!"
    os.mkdir(project_path)
    os.mkdir(project_path+configs['Model'])
    os.mkdir(project_path+configs['File'])
    os.mkdir(project_path+configs['Eval']['save_imgs'])
    
    # Save the dictionary as a yaml file
    with open(project_path+'/config.yaml', 'w') as yaml_file:
        yaml.dump(configs, yaml_file)

    print(len(hr_files), len(lr_files))
    train_dataset = supervisedIQT(configs, lr_files, hr_files)
    train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    hr_files_test = glob.glob(configs['Data']['groundtruth_path_test'])
    lr_files_test = glob.glob(configs['Data']['lowres_path_test'])
    if configs['Train']['batch_sample']:
        batch_size_test = 1
    else:
        batch_size_test = configs['Eval']['batch_size']

    print(len(hr_files_test), len(lr_files_test))
    valid_dataset = supervisedIQT(configs, lr_files_test, hr_files_test, train=False)
    valid_loader =  DataLoader(valid_dataset, batch_size=batch_size_test, shuffle=False, drop_last=False)
    data = next(iter(valid_loader))

    print(len(train_loader), len(valid_loader), data[0].shape, data[1].shape)
    
    min_bound = (0. - configs['Data']['mean'])/ configs['Data']['std']
    print("Min bound ", min_bound)
   
    if configs['Train']['batch_sample']:
        img_size = configs['Train']['patch_size_sub']*configs['Train']['batch_sample_factor']
    else:
        img_size = configs['Train']['patch_size_sub']
    #Load model
    # unet for imagen
    unet1 = NullUnet()
    print("Unet1 loaded")
    unet2 = SRUnet256(
        img_size = img_size,
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
        use_se_attn = configs['Train']['use_se'],
        pixel_shuffle_upsample = True,
        boundary = configs['Train']['boundary'],
        batch_sample = configs['Train']['batch_sample'],
        batch_sample_factor = configs['Train']['batch_sample_factor'],
        deep_feature = configs['Train']['deep_feature']
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
        medlpips = configs['Train']['medlpips'],
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
    if configs['Train']['pretrain']:
        trainer.load(configs['Train']['pretrain_model'], strict=False)
        print("Pretrained model is loaded")

    print("Trainer loaded")
    trainer.add_train_dataset(train_dataset, batch_size = batch_size)
    trainer.add_valid_dataset(valid_dataset, batch_size = batch_size_test)
    print(batch_size_test)
    print("Model and Data are loaded!")
   
    # # working training loop
    train_ls = []
    valid_ls = []
    ssim_val = []
    psnr_val = []
    best = 10000.0
    for i in range(10000):
        loss = trainer.train_step(unet_number = 2, max_batch_size = configs['Train']['batch_size'])
        train_ls.append(loss)
        train_loss_save = pd.DataFrame({'loss': train_ls}).to_csv(project_path+configs['File']+configs['Train']['save_file'], index=False)
        trainer.update(unet_number = 2)

        if (i % eval_step == 0):
            print(f'unet: 2, Step: {i*len(train_loader)}, loss: {loss}')
            
            valid_loss, preds, condi1, data, ssim, psnr = trainer.valid_step(unet_number = 2, max_batch_size = configs['Eval']['batch_size'])

            valid_loss = np.mean(valid_loss)
            valid_ls.append(valid_loss)
            ssim_val.append(ssim)
            psnr_val.append(psnr)

            if configs['Train']['pred_obj'] == 'x_start':
                valid_loss_save = pd.DataFrame({'loss': valid_ls, 'ssim': ssim_val, 'psnr': psnr_val}).to_csv(project_path+configs['File']+configs['Eval']['save_file'], index=False)
            else:
                valid_loss_save = pd.DataFrame({'loss': valid_ls}).to_csv(project_path+configs['File']+configs['Eval']['save_file'], index=False)
            if best > valid_loss:
                print("Best model!")
                best = valid_loss
                save_img = data[0] #gt
                save_img2 = data[1] #lowres
                save_img3 = condi1 #x_noisy
               
                j = i*len(train_loader)
                with open(project_path+configs['Eval']['save_imgs']+f'conditional_iqt_{i}_gt.npy', 'wb') as f:
                    np.save(f, save_img)
                with open(project_path+configs['Eval']['save_imgs']+f'conditional_iqt_{i}_lr.npy', 'wb') as f:
                    np.save(f, save_img2)
                with open(project_path+configs['Eval']['save_imgs']+f'conditional_iqt_{i}_noisy.npy', 'wb') as f:
                    np.save(f, save_img3)
                with open(project_path+configs['Eval']['save_imgs']+f'conditional_iqt_{i}_pred.npy', 'wb') as f:
                    np.save(f, preds)
                trainer.save(project_path+configs['Model']+configs['Train']['save_model'])

