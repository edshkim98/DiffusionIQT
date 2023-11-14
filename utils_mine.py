import numpy as np
import torch
from einops.layers.torch import Rearrange, Reduce
import random
import torch.nn.functional as F
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convertVolume2subVolume2(image, factor = 3):
    return  Rearrange('b c (p1 h) (p2 w) (p3 d) -> (b p1 p2 p3) c h w d', p1=factor,p2=factor,p3=factor)(image)

def merge_sub_volumes2(sub_volumes, factor = 3):
    return Rearrange('(b b1 b2 b3) c h w d -> b c (b1 h) (b2 w) (b3 d)', b1=factor, b2=factor, b3=factor)(sub_volumes)

def convertVolume2subVolume(image, target_shape=(27,1,32,32,32)):
    if len(image.shape) != 5 or len(target_shape) != 5:
        raise ValueError("Both input and target shapes must have 5 dimensions")
        
    _, C1, W, H, D = image.shape
    B, C2, A, _, _ = target_shape
    
    assert C1==C2,'channels are not same'

    split_w = int(W // A)
    split_h = int(H // A)
    split_d = int(D // A)
    
    if B != split_w * split_h * split_d:
        raise ValueError("The target batch size must be the product of split dimensions")
    
    sub_volumes = image.unfold(2, A, A).unfold(3, A, A).unfold(4, A, A).permute(0,4,3,2,1,5,6,7).reshape(B, C1, A, A, A)
    return sub_volumes

def merge_sub_volumes(sub_volumes, original_shape=(1,1,96,96,96)):
    if len(sub_volumes.shape) != 5 or len(original_shape) != 5:
        raise ValueError("Both input and target shapes must have 5 dimensions")

    B, C1, A, _, _ = sub_volumes.shape
    _, C2, W, H, D = original_shape

    assert C1==C2,'channels are not same'

    split_w = int(W // A)
    split_h = int(H // A)
    split_d = int(D // A)

    if B != split_w * split_h * split_d:
        print(B, split_w, split_h, split_d)
        raise ValueError("The batch size must be the product of split dimensions")

    sub_volumes_split = sub_volumes.reshape(split_w, split_h, split_d, C1, A, A, A)

    merge_d = torch.cat(tuple(sub_volumes_split), dim=-1)
    merge_h = torch.cat(tuple(merge_d), dim=-2)
    result = torch.cat(tuple(merge_h), dim=-3)
    
    return torch.unsqueeze(result,0)

def volume_to_slices(volume):
    """
    Convert a 3D volume to a series of 3-channel 2D slices.
    
    Args:
    - volume (torch.Tensor): A 3D tensor of shape (B, C, H, W, D).
    
    Returns:
    - slices (torch.Tensor): A 4D tensor of shape (D-2, 3, H, W).
    """
    # Initialize an empty list to store the slices
    slices = []
    
    # Iterate through the depth dimension
    cnt = 0
    for d in range(0,volume.shape[-1] - 2, 9):
        # Extract three consecutive slices and stack them along the channel dimension
        slice_3ch_coronal = torch.concat([volume[:,:,:,d], volume[:,:,:,d+1], volume[:,:,:,d+2]], dim=1)
        slice_3ch_saggital = torch.concat([volume[:,:,d], volume[:,:,d+1], volume[:,:,d+2]], dim=1)

        slice_3ch_coronal = (slice_3ch_coronal - slice_3ch_coronal.min()) / (slice_3ch_coronal.max() - slice_3ch_coronal.min())
        slice_3ch_saggital = (slice_3ch_saggital - slice_3ch_saggital.min()) / (slice_3ch_saggital.max() - slice_3ch_saggital.min())

        slice_3ch_coronal = F.interpolate(slice_3ch_coronal, size=(224), mode='bilinear', align_corners=False)
        slice_3ch_saggital = F.interpolate(slice_3ch_saggital, size=(224), mode='bilinear', align_corners=False)

        slices.append(slice_3ch_coronal)
        slices.append(slice_3ch_saggital)

    # Convert the list of slices into a 4D tensor
    slices = torch.concat(slices, dim=0)
    
    return slices