# =============================================================================
# coded by https://github.com/Whiax/
# =============================================================================
from torchvision.transforms import Normalize
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from os.path import join
import torch.nn as nn
import numpy as np
import torch
import pickle
import math
import datetime

# =============================================================================
# Dataset
# =============================================================================
# ImageNet 
class ImageNoiseDataset(Dataset):
    def __init__(self, imgs, transforms):
        self.imgs = imgs
        self.transforms = transforms
    
    def __len__(self): return len(self.imgs)
    
    def __getitem__(self, idx):
        dataset=self
        item = {}
        image = dataset.imgs[idx] / 255
        image = dataset.transforms(image)
        noise = torch.rand(image.shape)
        item['image'] = image
        item['noise'] = noise
        return item

# =============================================================================
# Model
# =============================================================================
# https://arxiv.org/abs/1910.03151 /  https://github.com/BangguWu/ECANet
class EcaModule(nn.Module):
    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1):
        super(EcaModule, self).__init__()
        assert kernel_size % 2 == 1
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.shape[0], 1, -1)
        y = self.conv(y)
        y = y.view(x.shape[0], -1, 1, 1).sigmoid()
        return x * y.expand_as(x)
#whiax
def Conv2d(*args, **kwargs):
    args = [int(a) if type(a) != tuple else a for i,a in enumerate(args) if i < 6]
    if not 'padding' in kwargs:
        k = args[2] if len(args) > 2 else (kwargs['kernel_size'] if 'kernel_size' in kwargs else kwargs['k'])
        k = (k,k) if type(k) != tuple else k
        pad = ((k[0] - 1) // 2,(k[1] - 1) // 2)
        kwargs['padding'] = pad
    return nn.Conv2d(*args, **kwargs, **{'padding_mode':'zeros'})
class convolution(nn.Module):
    def __init__(self, inp_dim, out_dim, k=3, stride=1, groups=1, bn=True, act=True, dilation=1, bias=True, **kwargs):
        super(convolution, self).__init__()
        self.conv = Conv2d(inp_dim, out_dim, k, stride=(stride, stride), bias=not bn and bias, groups=groups, dilation=dilation, **kwargs)
        self.bn   = nn.BatchNorm2d(out_dim) if bn else nn.Identity()
        self.activation = nn.ReLU(True) if act else nn.Identity()
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out
class convolution_att(convolution):
    def __init__(self, inp_dim, out_dim, k=3, stride=1, groups=1, bn=True, act=True, dilation=1, attention='eca'):
        super(convolution_att, self).__init__(inp_dim, out_dim, k, stride, groups, bn, act, dilation)
        self.attention = EcaModule(out_dim)
    def forward(self, x):
        out = super().forward(x)
        out = self.attention(out)
        return out


#denoiser / whiax
class DenoiserModel(nn.Module):
    def __init__(self, f=44, depth_start_mult=2, depth_mult=2, depth=3, downsample2in1=[1], layconv=convolution_att):
        super().__init__()
        model=self
        
        fs = [f] 
        curmult = depth_start_mult
        for i in range(1, depth+1):
            fs += [f*int(curmult)]
            curmult *= depth_mult
        model.upsample = nn.Upsample(scale_factor=2)
        
        #head
        model.layer_base = nn.Sequential(*[layconv(3, f, 3, 1)])
        #down
        model.layers_downsample = nn.ModuleList()
        for i in range(1, depth+1):
            if not i in downsample2in1:
                model.layers_downsample += [nn.Sequential(*[
                    layconv(fs[i-1], fs[i], 3, 1),
                    layconv(fs[i], fs[i], 3, 2) ])]
            else:
                model.layers_downsample += [nn.Sequential(*[
                    layconv(fs[i-1], fs[i], 3, 2)  ])]
        #att
        model.fcatt = nn.Sequential(*[
            nn.Linear(fs[-1], fs[-1]),
            nn.Sigmoid()])
        #up
        model.layers_upsample = nn.ModuleList()
        for _i in range(0, depth):
            i = -_i-1
            l = nn.ModuleList()
            l += [layconv(fs[i], fs[i], 3, 1)]
            l += [layconv(fs[i], fs[i-1], 1, 1)]
            model.layers_upsample += [l]
        model.layer_tail = nn.Sequential(*[layconv(f, 8, 3, 1), convolution_att(8, 3, 1, 1, bn=False, act=False)])
        # initialize_weights(self)
    
    #forward mod
    def forward(self, x):
        model=self
        if len(x.shape) == 3: x=x.view([1,*x.shape])
        base_x = x = model.layer_base(x)
        xdi0 = []
        for lay in model.layers_downsample:
            x = lay(x)
            xdi0 += [x]
        x = xdi0[-1] * model.fcatt(xdi0[-1].mean([2,3])).view([xdi0[-1].shape[0], xdi0[-1].shape[1], 1, 1])
        for i, lays in enumerate(model.layers_upsample):
            x = xdi0[-(i+1)] + lays[0](x)
            x = lays[1](x)
            x = model.upsample(x)
        x = base_x + x
        x = model.layer_tail(x)
        x = x.clip(0,1)
        return x

# =============================================================================
# Methods
# =============================================================================
#pytorch channel first to np channel last
def pt_to_np(tensor):
    return np.ascontiguousarray(tensor.permute(1,2,0).numpy())
    
#load object
def load_object(name, folder='.'):
    return pickle.load(open(join(folder, name + '.pickle'), 'rb'))

#show pt tensor
def plt_imshow_pt(t):
    if 'cuda' in str(t.device):
        plt.imshow(pt_to_np(t.cpu()).astype(np.uint8))
    else:
        plt.imshow(pt_to_np(t).astype(np.uint8))

#dict to plot
def plot_dict(d, l='', source=None, **kwargs):
    if source is None: source = plt
    if l != '':
        source.plot(d.keys(), d.values(), label=l, **kwargs)
        source.legend(loc="upper left")
    else:
        source.plot(d.keys(), d.values(), **kwargs)

#get batch
def get_batch(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter

#normalize img tensor
normalize_t = Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))

# Return a dated id for a file/folder
def get_id():
    date = datetime.datetime.now()
    return f'{date.year}_{date.month:02}_{date.day:02}_{date.hour:02}_{date.minute:02}_{date.second:02}'













