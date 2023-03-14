from pathlib import Path
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils
import torch.nn.functional as F
import t5
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from datasets.utils.file_utils import get_datasets_user_agent
import io
import urllib
from torch.utils.data.dataloader import default_collate
import nibabel as nib

USER_AGENT = get_datasets_user_agent()

# helpers functions

def exists(val):
    return val is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# dataset, dataloader, collator

def my_collate(batch):
    batch = list(filter(lambda x : x is not None, batch))
    return default_collate(batch)

class supervisedIQT(Dataset):
    def __init__(self, lr_files, hr_files, fake=False, train=True):
        self.lr_files = lr_files
        self.hr_files = hr_files
        self.patch_size = 32
        self.fake = fake
        self.ratio = 0.7
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
        
        if self.fake:
            self.lr = np.load(self.files_lr[idx])
            self.hr = np.load(self.files_lr[idx].replace('lr','hr'))
            self.lr = self.transform(self.lr)
            self.hr = self.transform(self.hr)
                
            sample_lr = torch.unsqueeze(self.lr, 0)
            sample_hr = torch.unsqueeze(self.hr, 0)
            return sample_hr, sample_lr
        self.lr = self.files_lr[idx]
        self.hr = self.lr.replace('lr_norm','T1w_acpc_dc_restore_brain_sim036T_4x_groundtruth_norm')
        
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
        
class IQTDataset(Dataset):
    def __init__(
        self,
        hr_files,
        lr_files,
        fake = False
    ):
        self.hrfiles = hr_files
        self.lrfiles = lr_files
        self.fake = fake
        
        assert len(self.hrfiles) == len(self.hrfiles), "Length should be same"
    
    def transform(self, img, size=(256,256)):
        return TF.resize(img, size)
        
    def normalize(self, img):
        img = (img-img.min())/(img.max()-img.min())
        return img
    
    def np2tensor(self, x, length, mode='2d'):
        x = torch.tensor(x)
        if mode == '2d':
            if length == 2:
                x = torch.unsqueeze(x,0)
            elif length == 3:
                x = torch.unsqueeze(x,0)
        else:
            if length == 3:
                x = torch.unsqueeze(x,0)
            elif length == 4:
                x = torch.unsqueeze(x,0)
        return x

    def __len__(self):
        return len(self.hrfiles)

    def __getitem__(self, idx):

        if not self.fake:
            hrfile = self.hrfiles[idx]
            lrfile = self.hrfiles[idx].replace('groundtruth_', 'lr_')
            
            hrimg = np.load(hrfile).astype(np.float32)
            hrimg = self.np2tensor(hrimg, len(hrimg.shape))
            hrimg = self.transform(hrimg)
            hrimg = self.normalize(hrimg)
            
            lrimg = np.load(lrfile).astype(np.float32)
            lrimg = self.np2tensor(lrimg, len(lrimg.shape))
            lrimg = self.transform(lrimg)
            lrimg = self.normalize(lrimg)

        else:
            hrimg = torch.randn(1,32,32,32)
            lrimg = torch.randn(1,32,32,32)    
        return hrimg, lrimg
    
class Collator:
    def __init__(self, image_size, url_label, text_label, image_label, name, channels):
        self.url_label = url_label
        self.text_label = text_label
        self.image_label = image_label
        self.download = url_label is not None
        self.name = name
        self.channels = channels
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])
    def __call__(self, batch):

        texts = []
        images = []
        for item in batch:
            try:
                if self.download:
                    image = self.fetch_single_image(item[self.url_label])
                else:
                    image = item[self.image_label]
                image = self.transform(image.convert(self.channels))
            except:
                continue

            text = t5.t5_encode_text([item[self.text_label]], name=self.name)
            texts.append(torch.squeeze(text))
            images.append(image)

        if len(texts) == 0:
            return None
        
        texts = pad_sequence(texts, True)

        newbatch = []
        for i in range(len(texts)):
            newbatch.append((images[i], texts[i]))

        return torch.utils.data.dataloader.default_collate(newbatch)

    def fetch_single_image(self, image_url, timeout=1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = Image.open(io.BytesIO(req.read())).convert('RGB')
        except Exception:
            image = None
        return image

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        convert_image_to_type = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        convert_fn = partial(convert_image_to, convert_image_to_type) if exists(convert_image_to_type) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

def get_images_dataloader(
    folder,
    *,
    batch_size,
    image_size,
    shuffle = True,
    cycle_dl = False,
    pin_memory = True
):
    ds = Dataset(folder, image_size)
    dl = DataLoader(ds, batch_size = batch_size, shuffle = shuffle, pin_memory = pin_memory)

    if cycle_dl:
        dl = cycle(dl)
    return dl
