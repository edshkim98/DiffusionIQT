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

    if batch == []:
        return None
    
    return default_collate(batch)

class supervisedIQT(Dataset):
    def __init__(self, config, lr_files, hr_files, train=True):
        self.config = config
        self.lr_files = lr_files
        self.hr_files = hr_files

        self.mean_lr = self.config['Data']['mean']#202.68109075616067 #35.493949511348724
        self.std_lr = self.config['Data']['std']#346.51374798642223 #37.11344433531084

        if self.config['Train']['batch_sample']:
            self.patch_size = self.config['Train']['patch_size_sub'] * self.config['Train']['batch_sample_factor']
        else:
            self.patch_size = self.config['Train']['patch_size_sub']
        self.train = train
        if self.train:
            self.ratio = 0.2
        else:
            self.ratio = 0.8

        self.files_lr = []
        self.files_hr = []
        
        for i in range(len(self.lr_files)):
            self.files_lr.append(self.lr_files[i])
            self.files_hr.append(self.hr_files[i])

    def __len__(self):
        return len(self.files_lr)

    def normalize(self, img, mode='lr'): # transform 3D array to tensor

        if self.config['Data']['norm'] == 'min-max':
            return 2*(((img-img.min())/(img.max()-img.min()))-0.5)
        return (img - self.mean_lr)/self.std_lr
    
    def cube(self,data):

        hyp_norm = data

        if len(hyp_norm.shape)>3:
            hyp_norm = hyp_norm[:,:, 2:258, 27:283]
        else:
            hyp_norm = hyp_norm[2:258, 27:283, :256]

        return hyp_norm

    def __getitem__(self, idx):
        
        self.lr = self.files_lr[idx]
        self.hr = self.lr.replace('lr_norm',self.config['Data']['groundtruth_fname'])#'T1w_acpc_dc_restore_brain')   #'T1w_acpc_dc_restore_brain_sim036T_4x_groundtruth_norm')
        
        self.lr = nib.load(self.lr)
        self.lr_affine = self.lr.affine
        self.lr = torch.tensor(self.lr.get_fdata().astype(np.float32))
        self.img_shape = self.lr.shape
        self.hr = nib.load(self.hr)
        self.hr_affine = self.hr.affine
        self.hr = torch.tensor(self.hr.get_fdata().astype(np.float32))
        
        #Cube
        low, high = 0, 256 #16, 240 
        
        assert self.lr.shape == (256,256,256), f'lr must be 256 256 256 but got {self.lr.shape}'
        assert self.hr.shape == (256,256,256), f'hr must be 256 256 256 but got {self.hr.shape}'

        self.lr = self.lr[low:high,low:high,low:high]
        self.hr = self.hr[low:high,low:high,low:high] 

        random_idx = np.random.randint(low=0, high=(high-low)-self.patch_size, size=3)
        self.lr = self.lr[random_idx[0]:random_idx[0]+self.patch_size, random_idx[1]:random_idx[1]+self.patch_size, random_idx[2]:random_idx[2]+self.patch_size]
        self.hr = self.hr[random_idx[0]:random_idx[0]+self.patch_size, random_idx[1]:random_idx[1]+self.patch_size, random_idx[2]:random_idx[2]+self.patch_size]
	
        non_zero = np.count_nonzero(self.lr)
        self.total_voxel = self.patch_size * self.patch_size * self.patch_size
        non_zero_proportion = (non_zero/self.total_voxel)
        if (non_zero_proportion < self.ratio):
            return self.__getitem__(idx)
                            
        self.lr = self.normalize(self.lr, mode='lr')
        self.hr = self.normalize(self.hr, mode='hr')
            
        sample_lr = torch.unsqueeze(self.lr, 0)
        sample_hr = torch.unsqueeze(self.hr, 0)
        
        if self.train:
            return sample_hr, sample_lr
        else:    
            return sample_hr, sample_lr
        
class supervisedIQT_INF(Dataset):
    def __init__(self, config, lr_file):
        self.lr_file = lr_file
        self.config = config

        self.mean_lr = self.config['Data']['mean']#202.68109075616067 #35.493949511348724
        self.std_lr = self.config['Data']['std']#346.51374798642223 #37.11344433531084

        if self.config['Train']['batch_sample']:
            self.patch_size = self.config['Train']['patch_size_sub'] * self.config['Train']['batch_sample_factor']
        else:
            self.patch_size = self.config['Train']['patch_size_sub']
        self.overlap = self.config['Eval']['overlap']

        self.ratio = 0.05
        self.total_voxel = self.patch_size * self.patch_size * self.patch_size

        self.lr_idx = []
        low, high = 0, 256
        self.lr_data = nib.load(self.lr_file).get_fdata()[low:high,low:high,low:high]
        for i in range(0,self.lr_data.shape[0]-self.patch_size+1,self.overlap):
            for j in range(0,self.lr_data.shape[1]-self.patch_size+1,self.overlap):
                for k in range(0,self.lr_data.shape[2]-self.patch_size+1,self.overlap):
                    self.lr_idx.append([i,j,k])

    def __len__(self):
        return len(self.lr_idx)

    def normalize(self, img): # transform 3D array to tensor

        image_torch = torch.FloatTensor(img)
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
        
        self.lr = self.lr_idx[idx]
        
        self.lr = self.lr_data[self.lr[0]:self.lr[0]+self.patch_size, self.lr[1]:self.lr[1]+self.patch_size, self.lr[2]:self.lr[2]+self.patch_size]
        self.lr = torch.tensor(self.lr.astype(np.float32))
        self.img_shape = self.lr.shape
            
        non_zero = np.count_nonzero(self.lr)
        non_zero_proportion = (non_zero/self.total_voxel)
        
        if (non_zero_proportion < self.ratio):
            return None 
                            
        self.lr = self.normalize(self.lr)
            
        sample_lr = torch.unsqueeze(self.lr, 0)
        
        return [sample_lr, torch.tensor(self.lr_idx[idx])]



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
