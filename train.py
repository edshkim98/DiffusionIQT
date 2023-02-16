import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import yaml

import cv2
from imagen_pytorch import Unet, NullUnet, Imagen, SRUnet256, alpha_cosine_log_snr
from data import IQTDataset
from trainer import ImagenTrainer

import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    with open('./config/config.yaml','r') as file:
        configs = yaml.safe_load(file)

    hr_files = glob.glob(configs['Data']['groundtruth_path'])
    lr_files = glob.glob(configs['Data']['lowres_path'])
    batch_size = configs['Train']['batch_size']

    print(len(hr_files), len(lr_files))
    train_dataset = IQTDataset(hr_files, lr_files)
    train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    hr_files_test = glob.glob(configs['Data']['groundtruth_path_test'])
    lr_files_test = glob.glob(configs['Data']['lowres_path_test'])
    batch_size_test = configs['Eval']['batch_size']

    print(len(hr_files_test), len(lr_files_test))
    valid_dataset = IQTDataset(hr_files_test, lr_files_test)
    valid_loader =  DataLoader(valid_dataset, batch_size=batch_size_test, shuffle=False, drop_last=False)

    #Load model
    # unet for imagen
    unet1 = NullUnet()

    unet2 = SRUnet256(
        dim = 32,
        dim_mults = (1, 2, 4, 8),
        channels=1,
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, True),
        layer_cross_attns = False,
        cond_on_text =False
    )

    imagen = Imagen(
        condition_on_text = False, 
        unets = (unet1, unet2),
        image_sizes = (64, 256),
        channels=1,
        timesteps = 1000,
        cond_drop_prob = 0.0
    ).to(device)

    trainer = ImagenTrainer(
        imagen = imagen,
        #cosine_decay_max_steps = len(train_loader)*10,
        split_valid_from_train = False # whether to split the validation dataset from the training
    )
    trainer.add_train_dataset(train_dataset, batch_size = batch_size)
    trainer.add_valid_dataset(valid_dataset, batch_size = batch_size_test)
    print("Model and Data are loaded!")

    # working training loop
    train_ls = []
    valid_ls = []
    lst_best = 0
    best = 100
    for i in range(20000):
        loss = trainer.train_step(unet_number = 2, max_batch_size = 2)
        train_ls.append(loss)
        train_loss_save = pd.DataFrame({'loss': train_ls}).to_csv(configs['Train']['save_file'], index=False)
        trainer.update(unet_number = 2)
        if not (i % 50):
            print(f'unet: 2, Epoch: {i}, loss: {loss}')
            valid_loss, preds, data = trainer.valid_step(unet_number = 2, max_batch_size = 2)
            valid_loss = np.mean(valid_loss)
            valid_ls.append(valid_loss)
            valid_loss_save = pd.DataFrame({'loss': valid_ls}).to_csv(configs['Eval']['save_file'], index=False)
            print(f'valid loss: {valid_loss}')
            if best > valid_loss:
                print("Best model!")
                best = valid_loss
                save_img = list(map(lambda img: list(map(T.ToPILImage(), img.unbind(dim = 0))), data[0].cpu()))
                save_img2 = list(map(lambda img: list(map(T.ToPILImage(), img.unbind(dim = 0))), data[1].cpu()))
                save_img[0][0].save(configs['Eval']['save_imgs']+f'conditional_iqt_{i}_gt.png')
                save_img2[0][0].save(configs['Eval']['save_imgs']+f'conditional_iqt_{i}_lr.png')
                for j in range(len(preds)):
                    preds[j][0][0].save(configs['Eval']['save_imgs']+f'conditional_iqt_{j}_pred.png')
                trainer.save(configs['Train']['save_model'])
    trainer.save(configs['Train']['save_last_model'])