import numpy as np
import torch
from torchmetrics import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.functional import peak_signal_noise_ratio

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def psnr_impl(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    concat = torch.cat((pred,target),dim=1)
    PIXEL_MAX = torch.max(concat)
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def PSNR(pred, target):
    pred = (pred-pred.min())/(pred.max()-pred.min())
    target = (target-target.min())/(target.max()-target.min())

    return peak_signal_noise_ratio(pred, target, data_range=1.0)

def SSIM(pred, target, kernel_size=3, data_range=None):
    if data_range is None:
        pred = (pred-pred.min())/(pred.max()-pred.min())
        target = (target-target.min())/(target.max()-target.min())
        ssim = StructuralSimilarityIndexMeasure(kernel_size=kernel_size, data_range=1.0)
    else:
        ssim = StructuralSimilarityIndexMeasure(kernel_size=kernel_size, data_range=data_range)
    return ssim(pred, target)

def MSSIM(pred, target):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
    return ms_ssim(pred, target)
