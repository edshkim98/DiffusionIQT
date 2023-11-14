import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from MedicalNet.model import generate_model
from MedicalNet.setting import parse_opts 
from collections import OrderedDict

class Variables():
    def __init__(self):
        self.gpu_id = [0]
        self.n_seg_classes = 1
        self.img_list = '' 
        self.n_epochs = 1
        self.no_cuda = True
        self.data_root = ''
        self.pretrain_path = './MedicalNet/pretrain/resnet_10_23dataset.pth'
        self.batch_size = 1
        self.num_workers = 0
        self.model_depth = 10
        self.resnet_shortcut = 'B'
        self.input_D = 32
        self.input_H = 32
        self.input_W = 32
        self.model = 'resnet'
        self.phase = 'test'
        
class MedPerceptualLoss(torch.nn.Module):
    def __init__(self, model, resize=True):
        super(MedPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(model.conv1.eval())
        blocks.append(model.bn1.eval())
        blocks.append(model.relu.eval()) ####
        blocks.append(model.maxpool.eval())
        
        blocks.append(model.layer1.eval()) ####
        blocks.append(model.layer2.eval()) ####
        # blocks.append(model.layer3.eval())
        # blocks.append(model.layer4.eval())

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.mean = 271.64814106698583 
        self.std = 377.117173547721 
        self.min = (0 - self.mean) / self.std

    def denorm(self, volume):

        return volume * self.std + self.mean
    
    def norm(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        # volume = self.denorm(volume)

        pixels = volume[volume > self.min]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = torch.normal(0, 1, size = volume.shape).cuda()
        out[volume == 0.] = out_random[volume == 0.]
        return out

    def forward(self, input, target, feature_layers=[0, 1], style_layers=[0, 1]):

        # input = self.norm2(input)
        # target = self.norm2(target)
        feature_layers = list(np.array(feature_layers) + 4)
        feature_layers.append(2)

        if self.resize:
            orig_size = input.shape[-1]
            input = self.transform(input, mode='trilinear', size=(orig_size*2), align_corners=False)
            target = self.transform(target, mode='trilinear', size=(orig_size*2), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class MedPercept(nn.Module):
    def __init__(self, resize=True):
        super(MedPercept, self).__init__()

        self.resize = resize
        self.sets = Variables()
        self.m = generate_model(self.sets)

        self.x = torch.load('./MedicalNet/pretrain/resnet_10_23dataset.pth')['state_dict']
        self.x2 = OrderedDict()
        for k, v in self.x.items():
            self.x2[k[7:]] = v

        self.model = self.m[0]
        self.model.load_state_dict(self.x2, strict=False)

        self.percept = MedPerceptualLoss(self.model.cuda(), resize=self.resize)

    def forward(self, input, target, feature_layers=[0, 1], style_layers=[]):
        if len(input.shape) == 3:
            input = torch.unsqueeze(torch.unsqueeze(input,0),0)
            target = torch.unsqueeze(torch.unsqueeze(target,0),0)

        return self.percept(input, target, feature_layers, style_layers)