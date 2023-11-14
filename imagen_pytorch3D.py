import sys
import math
import copy
from random import random
import numpy as np
from beartype.typing import List, Union
from beartype import beartype
from tqdm.auto import tqdm
from functools import partial, wraps
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch import nn, einsum
from torch.special import expm1
import torchvision.transforms as T

import kornia.augmentation as K

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce
from einops_exts import rearrange_many, repeat_many, check_shape

from imagen_video import Unet3D
from utils_mine import convertVolume2subVolume, merge_sub_volumes, volume_to_slices
import matplotlib.pyplot as plt

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch.autograd
from percept_loss import *

torch.autograd.set_detect_anomaly(True)
# helper functions

def boundary_pad(img, batch_sample_factor=3):
    b, c, h  = img.shape[0], img.shape[1], img.shape[2]
    A = h+2
    B = h
    big_patch_size = h*batch_sample_factor
    img = merge_sub_volumes(img, original_shape=(1,c,big_patch_size,big_patch_size,big_patch_size)) #1,C,96,96,96
    img = F.pad(img, (1,1,1,1,1,1), "constant") #1,C,98,98,98
    sub_volumes = img.unfold(2, A, B).unfold(3, A, B).unfold(4, A, B).permute(0,4,3,2,1,5,6,7).reshape(b, c, A, A, A)

    return sub_volumes

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def divisible_by(numer, denom):
    return (numer % denom) == 0

def first(arr, d = None):
    if len(arr) == 0:
        return d
    return arr[0]

def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)
    return inner

def min_max_norm(img):
    return (img-img.min())/(img.max()-img.min())

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(val, length = None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output

def cast_uint8_images_to_float(images):
    if not images.dtype == torch.uint8:
        return images
    return images / 255

def module_device(module):
    return next(module.parameters()).device

def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def pad_tuple_to_length(t, length, fillvalue = None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue,) * remain_length))

# helper classes

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

# tensor helpers

def log(t, eps: float = 1e-12):
    return torch.log(t.clamp(min = eps))

def l2norm(t):
    return F.normalize(t, dim = -1)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim

    if padding_dims <= 0:
        return t
    
    return t.view(*t.shape, *((1,) * padding_dims)) #bs c h w d

def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)

def resize_image_to(
    image,
    target_image_size,
    clamp_range = None,
    mode = 'nearest'
):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    out = F.interpolate(image, target_image_size, mode = mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out

def calc_all_frame_dims(
    downsample_factors: List[int],
    frames
):
    if not exists(frames):
        return (tuple(),) * len(downsample_factors)

    all_frame_dims = []

    for divisor in downsample_factors:
        assert divisible_by(frames, divisor)
        all_frame_dims.append((frames // divisor,))

    return all_frame_dims

def safe_get_tuple_index(tup, index, default = None):
    if len(tup) <= index:
        return default
    return tup[index]

# image normalization functions
# ddpms expect images to be in the range of -1 to 1

def normalize_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# gaussian diffusion with continuous time helper functions and classes
# large part of this was thanks to @crowsonkb at https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

@torch.jit.script
def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

@torch.jit.script
def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5) # not sure if this accounts for beta being clipped to 0.999 in discrete version

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

class GaussianDiffusionContinuousTimes(nn.Module):
    def __init__(self, *, noise_schedule, timesteps = 1000):
        super().__init__()

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.num_timesteps = timesteps

    def get_times(self, batch_size, noise_level, *, device):
        return torch.full((batch_size,), noise_level, device = device, dtype = torch.float32)
        

    def sample_random_times(self, batch_size, *, device):
        random_tensor_cpu =  torch.zeros((batch_size,)).float().uniform_(0, 1)
        random_tensor_gpu = random_tensor_cpu.to(device)
        return random_tensor_gpu
    
    def get_condition(self, times):
        return maybe(self.log_snr)(times)

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.num_timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times
    
    def get_sampling_timesteps_non_uniform(self, batch, *, device):
        large_timesteps=10000
        gamma = torch.tensor(10)
        times = torch.linspace(1., 0., large_timesteps)

        probs = torch.exp(-gamma * times).type(torch.float64)
        probs = probs / probs.sum()

        ts = torch.tensor(np.random.choice(times, self.num_timesteps, p=probs, replace=False))

        if torch.tensor([1.0]) not in ts:
            ts = torch.cat((ts, torch.tensor([1.0])))
        if torch.tensor([0.0]) not in ts:
            ts = torch.cat((ts, torch.tensor([0.0])))
            
        ts = torch.sort(ts,descending=True).values.to(device)
        ts = repeat(ts, 't -> b t', b = batch)
        ts = torch.stack((ts[:, :-1], ts[:, 1:]), dim = 0)
        ts = ts.unbind(dim = -1)

        return ts

    def q_posterior(self, x_start, x_t, t, *, t_next = None):
        #Calculates predicted mean and variance for x_s given x_t. Here x_s is x_start (not necessarily x_T)
        t_next = default(t_next, lambda: (t - 1. / self.num_timesteps).clamp(min = 0.))

        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(partial(right_pad_dims_to, x_t), (log_snr, log_snr_next))

        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)

        # following (eq. 33)
        posterior_variance = (sigma_next ** 2) * c
        posterior_log_variance_clipped = log(posterior_variance, eps = 1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise = None):
        dtype = x_start.dtype
        if isinstance(t, float):
            batch = x_start.shape[0]
            t = torch.full((batch,), t, device = x_start.device, dtype = dtype)

        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr(t).type(dtype)
        log_snr_padded_dim = right_pad_dims_to(x_start, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)

        return alpha * x_start + sigma * noise, log_snr, alpha, sigma

    def q_sample_from_to(self, x_from, from_t, to_t, noise = None):
        shape, device, dtype = x_from.shape, x_from.device, x_from.dtype
        batch = shape[0]

        if isinstance(from_t, float):
            from_t = torch.full((batch,), from_t, device = device, dtype = dtype)

        if isinstance(to_t, float):
            to_t = torch.full((batch,), to_t, device = device, dtype = dtype)

        noise = default(noise, lambda: torch.randn_like(x_from))

        log_snr = self.log_snr(from_t)
        log_snr_padded_dim = right_pad_dims_to(x_from, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)

        log_snr_to = self.log_snr(to_t)
        log_snr_padded_dim_to = right_pad_dims_to(x_from, log_snr_to)
        alpha_to, sigma_to =  log_snr_to_alpha_sigma(log_snr_padded_dim_to)

        return x_from * (alpha_to / alpha) + noise * (sigma_to * alpha - sigma * alpha_to) / alpha

    def predict_start_from_v(self, x_t, t, v):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return alpha * x_t - sigma * v

    def predict_start_from_noise(self, x_t, t, noise): 
        #generate x_t-1 from x_t
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / alpha.clamp(min = 1e-8)

# norms and residuals

class LayerNorm(nn.Module):
    def __init__(self, feats, stable = False, dim = -1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim = dim, keepdim = True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = dim, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = dim, keepdim = True)
        # print(self.g.shape, mean.shape, var.shape, x.shape)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)

ChanLayerNorm = partial(LayerNorm, dim = -4) # h c h w d

class Always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)

def Upsample(dim, dim_out = None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'trilinear'),
        nn.Conv3d(dim, dim_out, 3, padding = 1)
    )

class PixelShuffle3D(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)
    
class Deconv3D(nn.Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=2, padding=1):
        super(Deconv3D, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel, kernel, kernel),
                            stride=(stride, stride, stride), padding=(padding, padding, padding), output_padding=1, bias=True),
            nn.Mish())

    def forward(self, x):
        return self.deconv(x)

def Upsample_deconv(dim, dim_out = None):
    dim_out = default(dim_out, dim)

    return Deconv3D(dim, dim_out)

class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv3d(dim, dim_out * 8, 1, padding_mode='replicate')

        self.net = nn.Sequential(
            conv,
            nn.Mish(),
            PixelShuffle3D(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w, d = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w, d)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(dim, dim_out = None):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) (d s3) -> b (c s1 s2 s3) h w d', s1 = 2, s2 = 2, s3 = 2),
        nn.Conv3d(dim * 8, dim_out, 1)
    )

def Downsample2(dim, dim_out = None):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        nn.Conv3d(dim, dim_out, 3, padding = 1, stride=2)
    )

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1)

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        norm = True,
        boundary = False,
        factor = 3
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.Mish()
        self.boundary = boundary
        self.factor = factor
        if self.boundary:
            self.project = nn.Conv3d(dim, dim_out, 3)
        else:
            self.project = nn.Conv3d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.groupnorm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        if self.boundary:
            x = boundary_pad(x, batch_sample_factor=self.factor)

        return self.project(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        time_cond_dim = None,
        groups = 8,
        use_se = False,
        boundary = False,
        factor = 3
    ):
        super().__init__()

        self.time_mlp = None
        self.boundary = boundary
        self.factor = factor
                
        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.Mish(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )


        self.block1 = Block(dim, dim_out, groups = groups, boundary=self.boundary, factor = self.factor)
        self.block2 = Block(dim_out, dim_out, groups = groups, boundary=self.boundary, factor = self.factor)

        self.se = SE3D(dim_out, reduction=16) if use_se else Identity()

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else Identity()


    def forward(self, x, time_emb = None):
        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)
        
        h = self.block1(x)
        h = self.block2(h, scale_shift = scale_shift)
        #h = h * self.gca(h)
        h = self.se(h)
        
        out = h + self.res_conv(x)

        return out


class SE3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)
    
class GlobalContext(nn.Module):
    """ basically a superior form of squeeze-excitation that is attention-esque """

    def __init__(
        self,
        *,
        dim_in,
        dim_out
    ):
        super().__init__()
        self.to_k = nn.Conv3d(dim_in, 1, 1)
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv3d(dim_in, hidden_dim, 1),
            nn.Mish(),
            nn.Conv3d(hidden_dim, dim_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        x, context = rearrange_many((x, context), 'b n ... -> b n (...)')
        out = einsum('b i n, b c n -> b c i', context.softmax(dim = -1), x)
        out = rearrange(out, '... -> ... 1 1')
        return self.net(out)

class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv3d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim = 1)

class UpsampleCombiner(nn.Module):
    def __init__(
        self,
        dim,
        *,
        enabled = False,
        dim_ins = tuple(),
        dim_outs = tuple()
    ):
        super().__init__()
        dim_outs = cast_tuple(dim_outs, len(dim_ins))
        assert len(dim_ins) == len(dim_outs)

        self.enabled = enabled

        if not self.enabled:
            self.dim_out = dim
            return

        self.fmap_convs = nn.ModuleList([Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def forward(self, x, fmaps = None):
        target_size = x.shape[-1]

        fmaps = default(fmaps, tuple())

        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x

        fmaps = [resize_image_to(fmap, target_size) for fmap in fmaps]
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        return torch.cat((x, *outs), dim = 1)

################################################################################################
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_size: int = 256,
                 num_heads: int = 8,
                 dim_head: int = 64,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.0,
                 patch_num: int = 4,
                 local: bool=True):
        super().__init__()
        self.block = nn.Sequential(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads = num_heads, dropout = drop_p, dim_head= dim_head),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p, patch_num=patch_num, local=local),
                nn.Dropout(drop_p)
            ))
        )
    
    def forward(self, x):
        return self.block(x)
        
class TransformerEncoder(nn.Module):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(**kwargs) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0., patch_num: int = 4, local=False):
        super().__init__()

        if local:
            self.up_proj = nn.Sequential(
                                        Rearrange('b (h w d) c -> b c h w d', h=patch_num, w=patch_num, d=patch_num),
                                        nn.Conv3d(emb_size, emb_size*expansion, kernel_size=1),
                                        nn.Mish()
                                        )
            
            self.depth_conv = nn.Sequential(
                            depthwise_separable_conv3d(emb_size*expansion, emb_size*expansion, kernel_size=3, stride=1, padding=1),
                            nn.Mish()
                            )
            
            self.down_proj = nn.Sequential(
                                        nn.Conv3d(emb_size*expansion, emb_size, kernel_size=1),
                                        nn.Dropout(drop_p),
                                        Rearrange('b c h w d ->b (h w d) c')
                                        )
            self.net = nn.Sequential(
                self.up_proj,
                self.depth_conv,
                self.down_proj
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(emb_size, expansion * emb_size),
                nn.Mish(),
                nn.Dropout(drop_p),
                nn.Linear(expansion * emb_size, emb_size),
            )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 128, num_heads: int = 8, dim_head: int = 64, dropout: float = 0):
        super().__init__()

        self.mid_emb_size = emb_size
        self.emb_size = emb_size
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.inner_dim = dim_head * num_heads
        self.qkv = nn.Linear(self.mid_emb_size , self.inner_dim* 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(self.inner_dim , emb_size)
        self.scaling = self.dim_head ** -0.5
        
    def forward(self, x, mask = None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> qkv b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) * self.scaling
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        att = F.softmax(energy, dim=-1) 
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_size: int = 128, img_size: int = 224, reduction: bool = False):
        self.patch_size = patch_size
        super().__init__()
        
        self.projection = nn.Sequential(
            depthwise_separable_conv3d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size), #nn.Conv3d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) (d)-> b (h w d) e')
        )

        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **3, emb_size))

    def forward(self, x):
        x = self.projection(x)
        x += self.positions
        return x

class depthwise_separable_conv3d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0):
        super(depthwise_separable_conv3d, self).__init__()
        self.depthwise = nn.Conv3d(input_dim, input_dim, kernel_size=kernel_size, stride=stride, padding=padding,
                                                              groups=input_dim)
        self.pointwise = nn.Conv3d(input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        return x
    
class ViT3D(nn.Module):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                num_heads: int = 8,
                dim_head: int = 64,
                img_size: int = 224,
                depth: int = 1,
                drop_p: float = 0.1,
                forward_drop_p: float = 0.3,
                forward_expansion: int = 2,
                reduction: bool = False, #VERSION reduction is deprecated and it refers to version of self attention
                local: bool = True,
                groups: int = 1,
                **kwargs):
        super().__init__()
        self.reduction = reduction
        self.emb_size = in_channels
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, self.emb_size, img_size, reduction=self.reduction)
        self.transformer_encoder = TransformerEncoder(depth, emb_size=self.emb_size, num_heads=num_heads, dim_head=dim_head, patch_num=img_size//patch_size, drop_p = drop_p, forward_drop_p = forward_drop_p, forward_expansion = forward_expansion, local = local,**kwargs)

        # self.reconstruction = nn.Sequential(
        #     Rearrange('b (h w d) c ->b c (h w d)', h=img_size//patch_size, w=img_size//patch_size, d=img_size//patch_size),
        #     nn.Conv1d(in_channels, in_channels * (patch_size ** 3), kernel_size=1, groups = groups),
        #     Rearrange('b (c p1 p2 p3) (h w d) -> b c (h p1) (w p2) (d p3)', p1=patch_size, p2=patch_size, p3=patch_size, h=img_size//patch_size, w=img_size//patch_size, d=img_size//patch_size)
        # )
        self.reconstruction = nn.Sequential(
                nn.LayerNorm(in_channels),
                Rearrange('b (h w d) c ->b c h w d', h=img_size//patch_size, w=img_size//patch_size, d=img_size//patch_size),
                nn.Upsample(scale_factor=patch_size, mode='trilinear', align_corners=True),
                depthwise_separable_conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                ChanLayerNorm(in_channels)
        )

    def forward(self, x):

        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        out = self.reconstruction(x)
        return out
#######################################################################################################################

class Patchify(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_size: int = 128, img_size: int = 224, reduction: bool = False):
        self.patch_size = patch_size
        super().__init__()
        
        self.norm = ChanLayerNorm(in_channels)
        self.projection = depthwise_separable_conv3d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size) #nn.Conv3d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
        
    def forward(self, x):
        x = self.norm(x)
        x = self.projection(x)
        return x
    
class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 8,
        dropout = 0.05,
        context_dim = None,
        patch_size = 2, # it must be 4 times smaller than original patch (e.g. 2 for 8x8x8 patch)
        img_size = 48,
        patch = False,
        groups = 1,
        **kwargs
    ):
        super().__init__()
        self.patch = patch
        self.patch_size = patch_size
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.Mish()

        if self.patch:
            self.patch_embed = Patchify(in_channels=dim, patch_size=patch_size, emb_size=dim, img_size=img_size) # H*C*W*C*D
            self.reconstruct = nn.Sequential(
                #nn.Conv3d(dim, dim*(patch_size ** 3), kernel_size=1, groups = groups),
                nn.Upsample(scale_factor=patch_size, mode='trilinear', align_corners=True),
                depthwise_separable_conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
                # Rearrange('b (c p1 p2 p3) h w d -> b c (h p1) (w p2) (d p3)', p1=patch_size, p2=patch_size, p3=patch_size, h=(img_size//patch_size), 
                #           w=(img_size//patch_size), d=(img_size//patch_size)),
                ChanLayerNorm(dim)
            )
        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv3d(dim, inner_dim, 1, bias = False),
            nn.Conv3d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv3d(dim, inner_dim, 1, bias = False),
            nn.Conv3d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv3d(dim, inner_dim, 1, bias = False),
            nn.Conv3d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, inner_dim * 2, bias = False)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Conv3d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap, context = None):

        if self.patch:
            fmap = self.patch_embed(fmap)

        h, x, y, z= self.heads, *fmap.shape[-3:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = rearrange_many((q, k, v), 'b (h c) x y z-> (b h) (x y z) c', h = h)

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            ck, cv = rearrange_many((ck, cv), 'b n (h d) -> (b h) n d', h = h)
            k = torch.cat((k, ck), dim = -2)
            v = torch.cat((v, cv), dim = -2)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y z) d -> b (h d) x y z', h = h, x = x, y = y, z=z)

        out = self.nonlin(out)
        out =  self.to_out(out)
        if self.patch:
            out = self.reconstruct(out)
        return out

class SoftMaxAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 8,
        dropout = 0.05,
        context_dim = None,
        patch_size = 2, # it must be 4 times smaller than original patch (e.g. 2 for 8x8x8 patch)
        img_size = 48,
        patch = False,
        groups = 1,
        **kwargs
    ):
        super().__init__()
        self.patch = patch
        self.patch_size = patch_size
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.Mish()

        if self.patch:
            self.patch_embed = Patchify(in_channels=dim, patch_size=patch_size, emb_size=dim, img_size=img_size) # H*C*W*C*D
            self.reconstruct = nn.Sequential(
                #nn.Conv3d(dim, dim*(patch_size ** 3), kernel_size=1, groups = groups),
                nn.Upsample(scale_factor=patch_size, mode='trilinear', align_corners=True),
                depthwise_separable_conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
                # Rearrange('b (c p1 p2 p3) h w d -> b c (h p1) (w p2) (d p3)', p1=patch_size, p2=patch_size, p3=patch_size, h=(img_size//patch_size), 
                #           w=(img_size//patch_size), d=(img_size//patch_size)),
                ChanLayerNorm(dim)
            )
        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv3d(dim, inner_dim, 1, bias = False),
            nn.Conv3d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv3d(dim, inner_dim, 1, bias = False),
            nn.Conv3d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv3d(dim, inner_dim, 1, bias = False),
            nn.Conv3d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, inner_dim * 2, bias = False)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Conv3d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm(dim)
        )
    
    def forward(self, fmap, context = None):

        if self.patch:
            fmap = self.patch_embed(fmap)

        h, x, y, z= self.heads, *fmap.shape[-3:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = rearrange_many((q, k, v), 'b (h c) x y z-> (b h) (x y z) c', h = h)

        energy = torch.einsum('bqd, bkd -> bqk', q, k) * self.scale

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            ck, cv = rearrange_many((ck, cv), 'b n (h d) -> (b h) n d', h = h)
            k = torch.cat((k, ck), dim = -2)
            v = torch.cat((v, cv), dim = -2)

        att = F.softmax(energy, dim=-1) 

        out = einsum('b n d, b d e -> b n e', att, v)
        out = rearrange(out, '(b h) (x y z) d -> b (h d) x y z', h = h, x = x, y = y, z=z)

        out = self.nonlin(out)
        out =  self.to_out(out)
        if self.patch:
            out = self.reconstruct(out)
        return out
    
def ChanFeedForward(dim, mult = 2):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv3d(dim, hidden_dim, 1, bias = False),
        nn.GELU(),
        ChanLayerNorm(hidden_dim),
        nn.Conv3d(hidden_dim, dim, 1, bias = False)
    )

class LinearAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        context_dim = None,
        patch_size = 2,
        img_size = 48,
        patch = False,
        groups = 1,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.patch = patch
        self.img_size = img_size

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearAttention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim, patch_size = patch_size, img_size= self.img_size,
                                patch = self.patch, groups = groups),
                ChanFeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, context = None):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        
        return x
    
class SoftMaxAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        context_dim = None,
        patch_size = 2,
        img_size = 48,
        patch = False,
        groups = 1,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.patch = patch
        self.img_size = img_size

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SoftMaxAttention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim, patch_size = patch_size, img_size= self.img_size,
                                patch = self.patch, groups = groups),
                ChanFeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, context = None):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        
        return x

class Unet(nn.Module):
    def __init__(
        self,
        *,
        dim,
        img_size = 96,
        num_resnet_blocks = 1,
        cond_dim = None,
        learned_sinu_pos_emb_dim = 16,
        dim_mults=(1, 2, 4, 8),
        cond_images_channels = 0,
        channels = 3,
        channels_out = None,
        attn_dim_head = 64,
        attn_heads = 8,
        ff_mult = 2.,
        lowres_cond = False,                # for cascading diffusion - https://cascaded-diffusion.github.io/
        att_type = 'vit',
        attend_at_middle = True,            # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        attend_at_middle_depth = 1,
        attend_at_middle_heads = 8,
        attend_at_enc = True,
        attend_at_enc_depth = 1,
        attend_at_enc_heads = 8,
        att_drop = 0.1,
        att_forward_drop = 0.3,
        att_forward_expansion = 2,
        att_skip_scale = False,
        att_localvit = True,
        groups = 1,
        emb_size = 768,
        init_dim = 32,
        resnet_groups = 8,
        init_conv_kernel_size = 3,          # kernel size of initial conv, if not using cross embed
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        memory_efficient = False,
        init_conv_to_final_conv_residual = False,
        use_se_attn = True,
        scale_skip_connection = False,
        final_resnet_block = True,
        final_conv_kernel_size = 1,
        self_cond = False,
        combine_upsample_fmaps = False,      # combine feature maps from all upsample blocks, used in unet squared successfully
        pixel_shuffle_upsample = True,       # may address checkboard artifacts
        boundary = False,
        batch_sample = True,
        batch_sample_factor = 3,
        deep_feature = True
            ):
        super().__init__()

        # guide researchers
        self.att_type = att_type
        self.dim_head = attn_dim_head
        self.att_localvit = att_localvit
        self.skip_scale = att_skip_scale
        self.batch_sample = batch_sample
        self.att_drop = att_drop
        self.att_forward_drop = att_forward_drop
        self.att_forward_expansion = att_forward_expansion
        self.img_size = img_size
        self.batch_sample_factor = batch_sample_factor
        self.boundary = boundary
        self.num_groups = groups
        self.deep_feature = deep_feature
        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'

        if dim < 128:
            print_once('The base dimension of your u-net should ideally be no smaller than 128, as recommended by a professional DDPM trainer https://nonint.com/2022/05/04/friends-dont-let-friends-train-small-diffusion-models/')

        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        self._locals.pop('self', None)
        self._locals.pop('__class__', None)
               
        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        # (1) in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        # (2) in self conditioning, one appends the predict x0 (x_start)
        init_channels = channels * (1 + int(lowres_cond))
        init_dim = default(init_dim, dim)

        self.self_cond = self_cond

        # optional image conditioning

        self.has_cond_image = cond_images_channels > 0
        self.cond_images_channels = cond_images_channels

        init_channels += cond_images_channels

        # initial convolution

        if self.boundary:
            self.init_conv = CrossEmbedLayer(init_channels, dim_out = init_dim, kernel_sizes = init_cross_embed_kernel_sizes, stride = 1) if init_cross_embed else nn.Conv3d(init_channels, init_dim, init_conv_kernel_size)
        else:
            self.init_conv = CrossEmbedLayer(init_channels, dim_out = init_dim, kernel_sizes = init_cross_embed_kernel_sizes, stride = 1) if init_cross_embed else nn.Conv3d(init_channels, init_dim, init_conv_kernel_size, padding = init_conv_kernel_size // 2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.dims = dims
        self.in_out = in_out

        # time conditioning

        cond_dim = default(cond_dim, dim) #64
        time_cond_dim = dim * 4 #64*4=256

        # embedding time for log(snr) noise from continuous version

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1

        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
            nn.Mish()
        )

        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        # low res aug noise conditioning
        self.lowres_cond = lowres_cond

        # normalizations
        self.norm_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)
        self.text_to_cond = None

        num_layers = len(in_out)

        # resnet block klass

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers) #(2,2,2)
        resnet_groups = cast_tuple(resnet_groups, num_layers) #(8,8,8)

        resnet_klass = ResnetBlock

        assert len(resnet_groups) == num_layers, 'number of resnet groups must be equal to number of layers'
        # downsample klass

        downsample_klass = Downsample

        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes = cross_embed_downsample_kernel_sizes)

        # scale for resnet skip connections

        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        layer_params = [num_resnet_blocks, resnet_groups]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers

        skip_connect_dims = [] # keep track of skip connection dimensions
        img_sizes = []
        self.patch_size = 8 ########################################### 48 -> 16 -> 4

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)

            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet

            pre_downsample = None

            if memory_efficient:
                pre_downsample = downsample_klass(dim_in, dim_out)
                current_dim = dim_out
            
            img_sizes.append(self.img_size)

            # self.patch_size = self.img_size//3//2 ########################################### 48 -> 16 -> 4
            self.img_size_transformer = img_sizes[-1]

            if ind != len(in_out)-1:
                skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet

            post_downsample = None
            if not memory_efficient:
                post_downsample = downsample_klass(current_dim, dim_out) if not is_last else nn.Conv3d(current_dim, dim_out, 1) 
            else:
                post_downsample = nn.Conv3d(dim_out, dim_out, 1)
            
            if attend_at_enc[ind]:
                if self.att_type == 'vit':
                    transformer_enc = ViT3D(in_channels=current_dim, patch_size=self.patch_size, num_heads=attend_at_enc_heads[ind], dim_head = self.dim_head, img_size=self.img_size_transformer, depth=attend_at_enc_depth[ind], 
                                        forward_drop_p=self.att_forward_drop, drop_p=self.att_drop, forward_expansion = self.att_forward_expansion, reduction=False, local=self.att_localvit, groups=self.num_groups) 
                elif self.att_type == 'linear':
                    transformer_enc = LinearAttentionTransformerBlock(dim = current_dim, depth = attend_at_enc_depth[ind], heads = attend_at_enc_heads[ind], dim_head = self.dim_head, ff_mult = self.att_forward_expansion
                                                                    , patch_size = self.patch_size, img_size= self.img_size_transformer, patch = True, groups = self.num_groups) 
                else:
                    transformer_enc = SoftMaxAttentionTransformerBlock(dim = current_dim, depth = attend_at_enc_depth[ind], heads = attend_at_enc_heads[ind], dim_head = self.dim_head, ff_mult = self.att_forward_expansion
                                                                    , patch_size = self.patch_size, img_size= self.img_size_transformer, patch = True, groups = self.num_groups) 
            else:
                transformer_enc = None

            self.downs.append(nn.ModuleList([
                pre_downsample,
                ResnetBlock(current_dim, current_dim, time_cond_dim = time_cond_dim, groups = groups, use_se = use_se_attn, boundary= self.boundary, factor = self.batch_sample_factor),
                transformer_enc,
                nn.ModuleList([ResnetBlock(current_dim, current_dim, time_cond_dim = time_cond_dim, groups = groups, use_se = use_se_attn, boundary= self.boundary, factor = self.batch_sample_factor) for _ in range(layer_num_resnet_blocks)]),
                post_downsample
            ]))
            self.img_size = self.img_size //2
            if not is_last:
                self.patch_size = self.patch_size //2
        # middle layers
        
        if self.deep_feature:
            if attend_at_middle:
                mid_dim = dims[-1] 
            else:
                mid_dim = dims[-1]
            if self.att_type == 'linear':
                self.mid_attn = LinearAttentionTransformerBlock(dim = mid_dim, depth = attend_at_middle_depth, heads = attend_at_middle_heads, dim_head = self.dim_head,
                                                             ff_mult = self.att_forward_expansion, patch_size = self.patch_size, img_size= img_sizes[-1], patch = True, groups = self.num_groups) if attend_at_middle else None
            elif self.att_type == 'softmax':
                self.mid_attn = SoftMaxAttentionTransformerBlock(dim = mid_dim, depth = attend_at_middle_depth, heads = attend_at_middle_heads, dim_head = self.dim_head,
                                                             ff_mult = self.att_forward_expansion, patch_size = self.patch_size, img_size= img_sizes[-1], patch = True, groups = self.num_groups) if attend_at_middle else None
            else:
                self.mid_attn = ViT3D(in_channels=mid_dim, patch_size=self.patch_size, num_heads=attend_at_middle_heads, dim_head = self.dim_head, img_size=img_sizes[-1],
                                   depth=attend_at_middle_depth, forward_drop_p=self.att_forward_drop, drop_p=self.att_drop, forward_expansion=self.att_forward_expansion, reduction=False, local=self.att_localvit, groups=self.num_groups) if attend_at_middle else None 
            self.mid_block = ResnetBlock(mid_dim, mid_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1], boundary= self.boundary, factor = self.batch_sample_factor)
        else:
            mid_dim = dims[-1] 
            self.mid_block = ResnetBlock(mid_dim, mid_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1], boundary= self.boundary, factor = self.batch_sample_factor)
        # upsample klass

        upsample_klass = Upsample_deconv if not pixel_shuffle_upsample else PixelShuffleUpsample

        # upsampling layers

        upsample_fmap_dims = []
        for ind, ((dim_out, dim_in), layer_num_resnet_blocks, groups) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            if ind == 0:
                dim_in = mid_dim
            is_last = ind == (len(in_out) - 1)

            if not is_last:
                skip_connect_dim = skip_connect_dims.pop()

            upsample_fmap_dims.append(dim_in)

            self.ups.append(nn.ModuleList([
                upsample_klass(dim_in, dim_out) if not is_last else None,
                resnet_klass(dim_out + skip_connect_dim, dim_out, time_cond_dim = time_cond_dim, groups = groups, use_se = use_se_attn, boundary=self.boundary, factor = self.batch_sample_factor) if not is_last else resnet_klass(dim_in, dim_out, time_cond_dim = time_cond_dim, groups = groups, use_se = use_se_attn, boundary=self.boundary, factor = self.batch_sample_factor) ,
                nn.ModuleList([ResnetBlock(dim_out, dim_out, time_cond_dim = time_cond_dim, groups = groups, use_se = use_se_attn, boundary= self.boundary, factor = self.batch_sample_factor) for _ in range(layer_num_resnet_blocks)]),
            ]))

        # whether to combine feature maps from all upsample blocks before final resnet block out

        # self.upsample_combiner = UpsampleCombiner(
        #     dim = dim,
        #     enabled = combine_upsample_fmaps,
        #     dim_ins = upsample_fmap_dims,
        #     dim_outs = dim
        # )

        # whether to do a final residual from initial conv to the final resnet block out

        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = dim_out #self.upsample_combiner.dim_out + (dim if init_conv_to_final_conv_residual else 0)

        # final optional resnet block and convolution out
        self.final_res_block = ResnetBlock(final_conv_dim, dim, time_cond_dim = time_cond_dim, groups = resnet_groups[0], use_se = use_se_attn, boundary=self.boundary, factor = self.batch_sample_factor) if final_resnet_block else None

        final_conv_dim_in = dim if final_resnet_block else final_conv_dim

        self.final_conv = nn.Conv3d(final_conv_dim_in, self.channels_out, final_conv_kernel_size)
        
        
    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        channels,
        channels_out,
    ):
        if lowres_cond == self.lowres_cond and \
            channels == self.channels and \
            channels_out == self.channels_out:
            return self

        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            channels = channels,
            channels_out = channels_out,
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    # methods for returning the full unet config as well as its parameter state

    def to_config_and_state_dict(self):
        return self._locals, self.state_dict()

    # class method for rehydrating the unet from its config and state dict

    @classmethod
    def from_config_and_state_dict(klass, config, state_dict):
        unet = klass(**config)
        unet.load_state_dict(state_dict)
        return unet

    # methods for persisting unet to disk

    def persist_to_file(self, path):
        path = Path(path)
        path.parents[0].mkdir(exist_ok = True, parents = True)

        config, state_dict = self.to_config_and_state_dict()
        pkg = dict(config = config, state_dict = state_dict)
        torch.save(pkg, str(path))

    # class method for rehydrating the unet from file saved with `persist_to_file`

    @classmethod
    def hydrate_from_file(klass, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))

        assert 'config' in pkg and 'state_dict' in pkg
        config, state_dict = pkg['config'], pkg['state_dict']

        return Unet.from_config_and_state_dict(config, state_dict)

    # forward with classifier free guidance

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time_steps,
        time,
        *,
        lowres_cond_img = None,
        cond_images = None,
        self_cond = None,
        cond_drop_prob = 0.
    ):
        batch_size, device = x.shape[0], x.device
        
        # condition on self

        if self.self_cond:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, self_cond), dim = 1)
        # add low resolution conditioning, if present
        assert not (self.lowres_cond and not exists(lowres_cond_img)), 'low resolution conditioning image must be present'

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

        # condition on input image
        assert not (self.has_cond_image ^ exists(cond_images)), 'you either requested to condition on an image on the unet, but the conditioning image is not supplied, or vice versa'

        if exists(cond_images):
            assert cond_images.shape[1] == self.cond_images_channels, 'the number of channels on the conditioning image you are passing in does not match what you specified on initialiation of the unet'
            cond_images = resize_image_to(cond_images, x.shape[-1])
            x = torch.cat((cond_images, x), dim = 1)

        # initial convolution
        if self.boundary:
            x = boundary_pad(x)
        x = self.init_conv(x)

        # init conv residual

        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()

        # time conditioning
        time_hiddens = self.to_time_hiddens(time) #take time vector and convert it to embeddings
        # derive time tokens
        t = self.to_time_cond(time_hiddens) #change channel to same as convolution channel but no of channels same, refer to actual time embedding(?)

        # go through the layers of the unet, down and up
        hiddens = []
        last = len(self.downs)
        for i, (pre_downsample, init_block, attn_block, resnet_blocks, post_downsample) in enumerate(self.downs):
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t)

            if exists(attn_block):
                res = x

                B, C, H = x.shape[0], x.shape[1], x.shape[-1]
                x = merge_sub_volumes(x, original_shape=(1,C,H*self.batch_sample_factor,H*self.batch_sample_factor,H*self.batch_sample_factor))
                x = attn_block(x)
                x = convertVolume2subVolume(x, target_shape=(B,C,H,H,H))
                #assert self.batch_sample, 'batch sample must be true for this to work'
                #if self.skip_scale:
                #    scale = torch.clamp((-torch.log(time_steps[0]))**2, max=1.0).to(device)
                #    x *= scale

                x += res

            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
            if i != last - 1:
                hiddens.append(x)
            if exists(post_downsample):
                x = post_downsample(x)

            residual = x

        if self.deep_feature:
        #x = self.mid_block1(x, t)
            if exists(self.mid_attn):
                res = x

                B, C, H = x.shape[0], x.shape[1], x.shape[-1]
                x = merge_sub_volumes(x, original_shape=(1,C,H*self.batch_sample_factor,H*self.batch_sample_factor,H*self.batch_sample_factor))
                x = self.mid_attn(x)
                x = convertVolume2subVolume(x, target_shape=(B,C,H,H,H))
            
                #x2 = self.mid_block1(x, t)
                #x = torch.cat((x1, x2), dim = 1)

                x = self.mid_block(x, t)
            else:
                #x = self.mid_block1(x, t)
                x = self.mid_block(x, t)
            #residual connection
            #x +=residual

        add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim = 1)

        up_hiddens = []

        for upsample, init_block, resnet_blocks in self.ups:
            if exists(upsample):
                x = upsample(x)
                x = add_skip_connection(x)
            x = init_block(x, t)
            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)

            up_hiddens.append(x.contiguous())

        # whether to combine all feature maps from upsample blocks

        #x = self.upsample_combiner(x, up_hiddens)

        # final top-most residual if needed

        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim = 1)

        if exists(self.final_res_block):
            x = self.final_res_block(x, t)
        
        # if self.boundary:
        #     x = boundary_pad(x)

        out = self.final_conv(x)

        return out

# null unet

class NullUnet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lowres_cond = False
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))

    def cast_model_parameters(self, *args, **kwargs):
        return self

    def forward(self, x, *args, **kwargs):
        return x

# predefined unets, with configs lining up with hyperparameters in appendix of paper

class BaseUnet64(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 512,
            dim_mults = (1, 2, 3, 4),
            num_resnet_blocks = 3,
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = False
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

class SRUnet256(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 128,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 4, 8, 8),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = True
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

class SRUnet1024(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 128,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 4, 8, 8),
            layer_cross_attns = (False, False, False, True),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = True
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

# main imagen ddpm class, which is a cascading DDPM from Ho et al.

class Imagen(nn.Module):
    def __init__(
        self,
        unets,
        configs,
        *,
        image_sizes,                                # for cascading ddpm, image size at each stage
        min_bound = 0,
        channels = 3,
        timesteps = 1000,
        cond_drop_prob = 0.1,
        loss_type = 'l2',
        noise_schedules = 'cosine',
        pred_objectives = 'noise',
        lowres_noise_schedule = 'linear',
        lowres_sample_noise_level = 0.2,            # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
        per_sample_random_aug_noise_level = False,  # unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
        auto_normalize_img = False,                  # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
        p2_loss_weight_gamma = 0.5,                 # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time
        p2_loss_weight_k = 1,
        dynamic_thresholding = True,
        dynamic_thresholding_percentile = 0.95,     # unsure what this was based on perusal of paper
        only_train_unet_number = None,
        temporal_downsample_factor = 1,
        lpips = False,
        medlpips = False,
        boundary = False
    ):
        super().__init__()
        
        self.configs = configs
        self.medlpips = medlpips
        # loss
        self.boundary = boundary
        if lpips:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda()
        else:
            self.lpips = None
        if self.medlpips:
            assert self.medlpips, "MedLPIPIS is current not in use"
            #self.lpips = MedPercept(resize=False)
        else:
            self.lpips = None

        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()
        self.loss_type = loss_type
        self.loss_fn = loss_fn

        #Minimum value threshold
        self.min_bound = min_bound

        # conditioning hparams

        self.condition_on_text = False
        self.unconditional = not False

        # channels

        self.channels = channels

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet

        unets = cast_tuple(unets)
        num_unets = len(unets)

        # determine noise schedules per unet

        timesteps = cast_tuple(timesteps, num_unets)

        # make sure noise schedule defaults to 'cosine', 'cosine', and then 'linear' for rest of super-resoluting unets

        noise_schedules = cast_tuple(noise_schedules)
        noise_schedules = pad_tuple_to_length(noise_schedules, 2, 'cosine')
        noise_schedules = pad_tuple_to_length(noise_schedules, num_unets, 'linear')

        # construct noise schedulers

        noise_scheduler_klass = GaussianDiffusionContinuousTimes
        self.noise_schedulers = nn.ModuleList([])

        for timestep, noise_schedule in zip(timesteps, noise_schedules):
            noise_scheduler = noise_scheduler_klass(noise_schedule = noise_schedule, timesteps = timestep)
            self.noise_schedulers.append(noise_scheduler)

        # lowres augmentation noise schedule

        self.lowres_noise_schedule = GaussianDiffusionContinuousTimes(noise_schedule = lowres_noise_schedule)

        # ddpm objectives - predicting noise by default

        self.pred_objectives = cast_tuple(pred_objectives, num_unets)

        # construct unets

        self.unets = nn.ModuleList([])

        self.unet_being_trained_index = -1 # keeps track of which unet is being trained at the moment
        self.only_train_unet_number = only_train_unet_number

        for ind, one_unet in enumerate(unets):
            assert isinstance(one_unet, (Unet, Unet3D, NullUnet))
            is_first = ind == 0

            one_unet = one_unet.cast_model_parameters(
                lowres_cond = not is_first,
                channels = self.channels,
                channels_out = self.channels
            )

            self.unets.append(one_unet)

        # unet image sizes

        image_sizes = cast_tuple(image_sizes)
        self.image_sizes = image_sizes

        assert num_unets == len(image_sizes), f'you did not supply the correct number of u-nets ({len(unets)}) for resolutions {image_sizes}'

        self.sample_channels = cast_tuple(self.channels, num_unets)

        # temporal interpolation

        temporal_downsample_factor = cast_tuple(temporal_downsample_factor, num_unets)
        self.temporal_downsample_factor = temporal_downsample_factor
        assert temporal_downsample_factor[-1] == 1, 'downsample factor of last stage must be 1'
        assert all([left >= right for left, right in zip((1, *temporal_downsample_factor[:-1]), temporal_downsample_factor[1:])]), 'temporal downssample factor must be in order of descending'

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (num_unets - 1))), 'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'

        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.per_sample_random_aug_noise_level = per_sample_random_aug_noise_level

        # classifier free guidance

        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.

        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity

        # dynamic thresholding

        self.dynamic_thresholding = cast_tuple(dynamic_thresholding, num_unets)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        # p2 loss weight

        self.p2_loss_weight_k = p2_loss_weight_k
        self.p2_loss_weight_gamma = cast_tuple(p2_loss_weight_gamma, num_unets)

        assert all([(gamma_value <= 2) for gamma_value in self.p2_loss_weight_gamma]), 'in paper, they noticed any gamma greater than 2 is harmful'

        # one temp parameter for keeping track of device

        self.register_buffer('_temp', torch.tensor([0.]), persistent = False)

        # default to device of unets passed in

        self.to(next(self.unets.parameters()).device)

    def force_unconditional_(self):
        self.condition_on_text = False
        self.unconditional = True

        for unet in self.unets:
            unet.cond_on_text = False

    @property
    def device(self):
        return self._temp.device

    def get_unet(self, unet_number):
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.unets]
            delattr(self, 'unets')
            self.unets = unets_list

        if index != self.unet_being_trained_index:
            for unet_index, unet in enumerate(self.unets):
                unet.to(self.device if unet_index == index else 'cpu')

        self.unet_being_trained_index = index
        return self.unets[index]

    def reset_unets_all_one_device(self, device = None):
        device = default(device, self.device)
        self.unets = nn.ModuleList([*self.unets])
        self.unets.to(device)

        self.unet_being_trained_index = -1

    @contextmanager
    def one_unet_in_gpu(self, unet_number = None, unet = None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        devices = [module_device(unet) for unet in self.unets]
        self.unets.cpu()
        unet.to(self.device)

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    # overriding state dict functions

    def state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # gaussian diffusion methods

    def p_mean_variance(
        self,
        unet,
        x,
        t,
        *,
        noise_scheduler,
        cond_images = None,
        lowres_cond_img = None,
        self_cond = None,
        cond_scale = 1.,
        model_output = None,
        t_next = None,
        pred_objective = 'noise',
        dynamic_threshold = True
    ):
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'imagen was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        pred = default(model_output, lambda: unet.forward_with_cond_scale(x, t, noise_scheduler.get_condition(t), cond_images = cond_images, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, self_cond = self_cond)) 
        
        if pred_objective == 'noise':
            #Calculates prediction x_t-1 from the noise predicted given x_t
            x_start = noise_scheduler.predict_start_from_noise(x, t = t, noise = pred)
        elif pred_objective == 'x_start':
            x_start = pred
        elif pred_objective == 'v':
            x_start = noise_scheduler.predict_start_from_v(x, t = t, v = pred)
        else:
            raise ValueError(f'unknown objective {pred_objective}')
        
        if dynamic_threshold:      

            # following pseudocode in appendix
            # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
            s = torch.quantile(
               rearrange(x_start, 'b ... -> b (...)').abs(),
               self.dynamic_thresholding_percentile,
               dim = -1
            )
            
            if self.configs['Data']['norm'] == 'min-max':
                s.clamp_(min=1.)
            else:
                s.clamp_(min = self.min_bound) #1.)
            s = right_pad_dims_to(x_start, s)
            x_start =  x_start.clamp(-s, s) / s
        else:
            if self.configs['Data']['norm'] == 'min-max':
                x_start.clamp_(-1., 1.)
            else:
                x_start.clamp_(min = self.min_bound)#(-1., 1.)
        
        #Deduce mean and variance
        mean_and_variance = noise_scheduler.q_posterior(x_start = x_start, x_t = x, t = t, t_next = t_next)
        return mean_and_variance, x_start

    @torch.no_grad()
    def p_sample(
        self,
        unet,
        x,
        t,
        *,
        noise_scheduler,
        t_next = None,
        cond_images = None,
        cond_scale = 1.,
        self_cond = None,
        lowres_cond_img = None,
        pred_objective = 'noise',
        dynamic_threshold = True
    ):
        b, *_, device = *x.shape, x.device

        (model_mean, _, model_log_variance), x_start = self.p_mean_variance(unet, x = x, t = t, t_next = t_next, noise_scheduler = noise_scheduler, cond_images = cond_images, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, self_cond = self_cond, pred_objective = pred_objective, dynamic_threshold = dynamic_threshold)
        noise = torch.randn_like(x)
        # no noise when t == 0, if t_next = 0 then nonzero mask is 0 hence pred = model_mean. 
        is_last_sampling_timestep = (t_next == 0) if isinstance(noise_scheduler, GaussianDiffusionContinuousTimes) else (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise #q_sampled images
        return pred, x_start #q_sampled images, predicted x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        unet,
        shape,
        *,
        noise_scheduler,
        lowres_cond_img = None,
        cond_images = None,
        inpaint_images = None,
        inpaint_masks = None,
        inpaint_resample_times = 5,
        init_images = None,
        skip_steps = None,
        cond_scale = 1,
        pred_objective = 'noise',
        dynamic_threshold = True,
        use_tqdm = True
    ):
        device = self.device

        batch = shape[0]
        img = torch.randn(shape, device = device)

        # for initialization with an image or video

        if exists(init_images):
            img += init_images

        # keep track of x0, for self conditioning

        x_start = None

        # prepare inpainting

        has_inpainting = exists(inpaint_images) and exists(inpaint_masks)
        resample_times = inpaint_resample_times if has_inpainting else 1

        # time

        #timesteps = noise_scheduler.get_sampling_timesteps_non_uniform(batch, device = device)
        timesteps = noise_scheduler.get_sampling_timesteps(batch, device = device)

        # whether to skip any steps

        skip_steps = default(skip_steps, 0)
        #timesteps = tuple(list(timesteps)[::skip_steps]) #timesteps[skip_steps:]
        if skip_steps > 1:
            timesteps = list(timesteps)[::skip_steps] + [list(timesteps)[-1]]
            timesteps = tuple(timesteps)        

        noisy_pred_img = []
        pred_img = []
        cnt = 0

        for times, times_next in tqdm(timesteps, desc = 'sampling loop time step', total = len(timesteps), disable = not use_tqdm):
            is_last_timestep = times_next == 0

            for r in reversed(range(resample_times)):
                is_last_resample_step = r == 0

                if has_inpainting:
                    noised_inpaint_images, *_ = noise_scheduler.q_sample(inpaint_images, t = times)
                    img = img * ~inpaint_masks + noised_inpaint_images * inpaint_masks

                self_cond = x_start if unet.self_cond else None

                img, x_start = self.p_sample(
                    unet,
                    img,
                    times,
                    t_next = times_next,
                    cond_images = cond_images,
                    cond_scale = cond_scale,
                    self_cond = self_cond,
                    lowres_cond_img = lowres_cond_img,
                    noise_scheduler = noise_scheduler,
                    pred_objective = pred_objective,
                    dynamic_threshold = dynamic_threshold
                )

                if has_inpainting and not (is_last_resample_step or torch.all(is_last_timestep)):
                    renoised_img = noise_scheduler.q_sample_from_to(img, times_next, times)

                    img = torch.where(
                        self.right_pad_dims_to_datatype(is_last_timestep),
                        img,
                        renoised_img
                    )
            if cnt % 1 == 0:
                noisy_pred_img.append(img.cpu().numpy())
                pred_img.append(x_start.cpu().numpy())
            cnt+=1
        
        noisy_pred_img.append(img.cpu().numpy())
        pred_img.append(x_start.cpu().numpy())
        if self.configs['Data']['norm'] == 'min-max':
            img.clamp_(-1.,1.)
        else:
            img.clamp_(min = self.min_bound) #-1., 1.)

        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img, noisy_pred_img, pred_img

    @torch.no_grad()
    @eval_decorator
    @beartype
    def sample(
        self,
        text_masks = None,
        text_embeds = None,
        video_frames = None,
        cond_images = None,
        inpaint_images = None,
        inpaint_masks = None,
        inpaint_resample_times = 5,
        init_images = None,
        skip_steps = None,
        batch_size = 1,
        cond_scale = 1.,
        lowres_sample_noise_level = None,
        start_at_unet_number = 1,
        start_image_or_video = None, #lowres_imgae input
        stop_at_unet_number = None,
        return_all_outputs = False,
        return_pil_images = False,
        device = None,
        use_tqdm = True
    ):
        device = default(device, self.device)
        self.reset_unets_all_one_device(device = device)

        cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        if not self.unconditional:
            assert exists(text_embeds), 'text must be passed in if the network was not trained without text `condition_on_text` must be set to `False` when training'

            text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim = -1))
            batch_size = text_embeds.shape[0]

        outputs = []

        is_cuda = next(self.parameters()).is_cuda
        device = next(self.parameters()).device

        lowres_sample_noise_level = default(lowres_sample_noise_level, self.lowres_sample_noise_level)

        num_unets = len(self.unets)

        # condition scaling

        cond_scale = cast_tuple(cond_scale, num_unets)

        # for initial image and skipping steps

        init_images = cast_tuple(init_images, num_unets)
        init_images = [init_image for init_image in init_images]

        skip_steps = cast_tuple(skip_steps, num_unets)

        # handle starting at a unet greater than 1, for training only-upscaler training

        if start_at_unet_number > 1:
            assert start_at_unet_number <= num_unets, 'must start a unet that is less than the total number of unets'
            assert not exists(stop_at_unet_number) or start_at_unet_number <= stop_at_unet_number
            assert exists(start_image_or_video), 'starting image or video must be supplied if only doing upscaling'

            img = start_image_or_video

        # go through each unet in cascade

        for unet_number, unet, channel, image_size, noise_scheduler, pred_objective, dynamic_threshold, unet_cond_scale, unet_init_images, unet_skip_steps in tqdm(zip(range(1, num_unets + 1), self.unets, self.sample_channels, self.image_sizes, self.noise_schedulers, self.pred_objectives, self.dynamic_thresholding, cond_scale, init_images, skip_steps), disable = not use_tqdm):

            if unet_number < start_at_unet_number:
                continue

            assert not isinstance(unet, NullUnet), 'one cannot sample from null / placeholder unets'

            context = self.one_unet_in_gpu(unet = unet) if is_cuda else nullcontext()

            with context:
                lowres_cond_img = None
                shape = (batch_size, channel, image_size, image_size)

                if unet.lowres_cond:
                    lowres_cond_img = img 

                shape = (batch_size, self.channels, image_size, image_size, image_size)
                
                img, lst_pred_noisy, lst_pred = self.p_sample_loop(
                    unet,
                    shape,
                    cond_images = cond_images,
                    inpaint_images = inpaint_images,
                    inpaint_masks = inpaint_masks,
                    inpaint_resample_times = inpaint_resample_times,
                    init_images = unet_init_images,
                    skip_steps = unet_skip_steps,
                    cond_scale = unet_cond_scale,
                    lowres_cond_img = lowres_cond_img,
                    noise_scheduler = noise_scheduler,
                    pred_objective = pred_objective,
                    dynamic_threshold = dynamic_threshold,
                    use_tqdm = use_tqdm
                )

                outputs.append(img)

            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        output_index = -1 if not return_all_outputs else slice(None) # either return last unet output or all unet outputs

        # if not return_pil_images:
        #     return outputs[output_index]

        return outputs[output_index], lst_pred_noisy, lst_pred

    @beartype
    def p_losses(
        self,
        unet: Union[Unet, Unet3D, NullUnet, DistributedDataParallel],
        x_start,
        times,
        *,
        noise_scheduler,
        lowres_cond_img = None,
        cond_images = None,
        noise = None,
        pred_objective = 'noise',
        p2_loss_weight_gamma = 0.,
        **kwargs
    ):

        noise = default(noise, lambda: torch.randn_like(x_start))

        # normalize to [-1, 1]
        x_start = self.normalize_img(x_start)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # get x_t
        x_noisy, log_snr, alpha, sigma = noise_scheduler.q_sample(x_start = x_start, t = times, noise = noise)

        # also noise the lowres conditioning image
        # at sample time, they then fix the noise level of 0.1 - 0.3
        lowres_cond_img_noisy = None
        lowres_cond_img_noisy = lowres_cond_img

        # time condition
        noise_cond = noise_scheduler.get_condition(times)

        # unet kwargs

        unet_kwargs = dict(
            cond_images = cond_images,
            lowres_cond_img = lowres_cond_img_noisy,
            cond_drop_prob = self.cond_drop_prob,
            **kwargs
        )

        # self condition if needed

        # Because 'unet' can be an instance of DistributedDataParallel coming from the
        # ImagenTrainer.unet_being_trained when invoking ImagenTrainer.forward(), we need to
        # access the member 'module' of the wrapped unet instance.
        self_cond = unet.module.self_cond if isinstance(unet, DistributedDataParallel) else unet.self_cond
        
        
        if self_cond and random() < 0.5:
            with torch.no_grad():
                pred = unet.forward(
                    x_noisy,
                    noise_cond,
                    **unet_kwargs
                ).detach()

                x_start = noise_scheduler.predict_start_from_noise(x_noisy, t = times, noise = pred) if pred_objective == 'noise' else pred

                unet_kwargs = {**unet_kwargs, 'self_cond': x_start}

        # get prediction
        pred = unet.forward(
            x_noisy,
            times, 
            noise_cond,
            **unet_kwargs
        )

        # prediction objective

        if pred_objective == 'noise':
            target = noise
        elif pred_objective == 'x_start':
            target = x_start
        elif pred_objective == 'v':
            # derivation detailed in Appendix D of Progressive Distillation paper
            # https://arxiv.org/abs/2202.00512
            # this makes distillation viable as well as solve an issue with color shifting in upresoluting unets, noted in imagen-video
            target = alpha * noise - sigma * x_start
        else:
            raise ValueError(f'unknown objective {pred_objective}')
        
        # losses
        if pred_objective == 'x_start':
            pred.clamp_(min = self.min_bound)
        losses = self.loss_fn(pred, target, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        
        # p2 loss reweighting

        if p2_loss_weight_gamma > 0:
            loss_weight = (self.p2_loss_weight_k + log_snr.exp()) ** -p2_loss_weight_gamma
            losses = losses * loss_weight

        if self.lpips:
            if self.medlpips:
                per_ls = self.lpips(pred, target)
            else:
                pred_rgb = volume_to_slices(pred)
                target_rgb = volume_to_slices(target)

                # pred_rgb = (pred_rgb - pred_rgb.min())/(pred_rgb.max() - pred_rgb.min())
                # target_rgb = (target_rgb - target_rgb.min())/(target_rgb.max() - target_rgb.min())

                per_ls = self.lpips(pred_rgb, target_rgb)


            return losses.mean() + 0.1*per_ls.mean(), pred, x_noisy, lowres_cond_img_noisy

        return losses.mean(), pred, x_noisy, lowres_cond_img_noisy

    @beartype
    def forward(
        self,
        images, # rename to images or video
        lowres_img = None,
        unet: Union[Unet, Unet3D, NullUnet, DistributedDataParallel] = None,
        text_embeds = None,
        text_masks = None,
        unet_number = None,
        cond_images = None,
        **kwargs
    ):

        assert images.shape[-1] == images.shape[-2], f'the images you pass in must be a square, but received dimensions of {images.shape[2]}, {images.shape[-1]}'
        assert not (len(self.unets) > 1 and not exists(unet_number)), f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, 'you can only train on unet #{self.only_train_unet_number}'

        images = cast_uint8_images_to_float(images)
        cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        assert images.dtype == torch.float, f'images tensor needs to be floats but {images.dtype} dtype found instead'

        unet_index = unet_number - 1

        unet = default(unet, lambda: self.get_unet(unet_number))

        assert not isinstance(unet, NullUnet), 'null unet cannot and should not be trained'

        noise_scheduler      = self.noise_schedulers[unet_index]
        p2_loss_weight_gamma = self.p2_loss_weight_gamma[unet_index]
        pred_objective       = self.pred_objectives[unet_index]
        target_image_size    = self.image_sizes[unet_index]
        
        b, c, *_, h, w, d, device = *images.shape, images.device

        check_shape(images, 'b c h w d', c = self.channels)
        assert h >= target_image_size and w >= target_image_size
        
        if self.configs['Train']['batch_sample']:
            times = noise_scheduler.sample_random_times(1, device = device)
            times = times.repeat(b)
        else:
            times = noise_scheduler.sample_random_times(b, device = device)

        if not self.unconditional:
            text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim = -1))

        lowres_cond_img = None
        assert lowres_img is not None, 'lowres image must be provided'
        lowres_cond_img = lowres_img
        self.lowres_cond_img = lowres_cond_img
        
        return self.p_losses(unet, images, times, cond_images = cond_images, noise_scheduler = noise_scheduler, lowres_cond_img = lowres_cond_img, pred_objective = pred_objective, p2_loss_weight_gamma = p2_loss_weight_gamma, **kwargs)
