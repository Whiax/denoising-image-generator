# =============================================================================
# coded by https://github.com/Whiax/, cite if 
# =============================================================================

# =============================================================================
# Imports
# =============================================================================
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T
from os.path import exists, join
import matplotlib.pyplot as plt
from methods import *
import numpy as np
import torch
import shutil
import time
import os
import timm
from collections import Counter

runif = np.random.uniform

import lpips
import lion




# =============================================================================
# Init
# =============================================================================
n_iter = 200000
img_size = 64
max_runtime = 2*60*60
#hyp
optim_kwargs = {'lr':7e-4} 
batch_size = 44
first_step = 0.94
second_step = 0.86
step_bef_0 = 0.4
n_other_steps = 3
start_loss_mult = 20
loss_restore_mult = 0
loss_preserve_mult = 0
n_features = 56
depth_start_mult=2
depth_mult=2.5
depth=4
non_fullrebuildnoise_step_loss = 3
downsample2in1 = [1]
layconv = convolution_att

assert n_other_steps >= 2
# =============================================================================
# Data
# =============================================================================
n_workers = 0
kwargs_dataloader = {'num_workers':n_workers,  'shuffle':True,  'drop_last':True,  'pin_memory':True, 'batch_size':batch_size}
transforms = T.Compose([T.RandomHorizontalFlip()])
imgs = load_object('dogs')
dataset = ImageNoiseDataset(imgs, transforms)
dataloader = DataLoader(dataset, **kwargs_dataloader)

#define noise rates
noise_rates = [1, first_step, second_step]
intermediate_noise_steps = list(np.linspace(second_step-0.1, step_bef_0, n_other_steps-1))
noise_rates += intermediate_noise_steps
noise_rates += [0]
noise_rates = list(reversed(sorted(noise_rates)))

# loss_mults = [20, 10, 5, 5, 5, 1]
loss_mults = [start_loss_mult] + [start_loss_mult/2] + [start_loss_mult/4]*(n_other_steps) + [1]
loss_mults = [max(1,e) for e in loss_mults]
loss_mults = [e/sum(loss_mults) for e in loss_mults]

#blur f
blur_f = lambda x: FT.gaussian_blur(x, 3, 3)
def damage_f(x):
    x = blur_f(x) if runif() < 0.4 else x
    x = FT.affine(x, runif(-20,20), [runif(-5,5),runif(-5,5)], 1, [runif(-20,20),runif(-20,20)]) if runif() < 0.9 else x
    x = FT.posterize((x*255).to(torch.uint8), 1)/255 if runif() < 0.2 else x
    return x

# =============================================================================
# Model
# =============================================================================
model = DenoiserModel(n_features, depth_start_mult, depth_mult, depth, downsample2in1, layconv)
model = model.cuda()

#external eval
imagenet_classifier = timm.create_model('xcit_tiny_12_p16_384_dist', pretrained=True)

# =============================================================================
# Optim
# =============================================================================
optimizer = lion.Lion(model.parameters(), **optim_kwargs)

#perceptual loss
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
loss_fn = loss_fn_vgg

# =============================================================================
# Misc
# =============================================================================
#clean logs
id_run = get_id()
log_dir = f'./log_{id_run}'
def get_log_dir():
    if not exists(log_dir):
        os.mkdir(log_dir)
        for p in ['methods', 'train']:
            p = p+'.py'
            shutil.copy(p, join(log_dir, '_'+p))
    return log_dir

# =============================================================================
# Loop
# =============================================================================
model.train()
start_iter = 0
it = start_iter
train_losses, val_scores = {}, {}
time_to_valscore = {}
batch_iter = iter(dataloader)
scaler = torch.cuda.amp.GradScaler()
start_loop_time = last_it_time = time.time()
for it in range(it, n_iter+1):
    batch, batch_iter = get_batch(dataloader, batch_iter)
    batch['image'] = batch['image'].cuda(non_blocking=True)
    batch['noise'] = batch['noise'].cuda(non_blocking=True)
    
    #infer
    with torch.cuda.amp.autocast():
        loss = 0
        #denoise task
        for i_nr, noise_rate in enumerate(noise_rates[:-1]):
            if i_nr <= non_fullrebuildnoise_step_loss:
                noisy_image = noise_rate * batch['noise'] + (1-noise_rate) * batch['image']
            else:
                noisy_image = denoised_pred
            denoised_truth = noise_rates[i_nr+1] * batch['noise'] + (1-noise_rates[i_nr+1]) * batch['image']
            denoised_pred = model(noisy_image)
            loss += loss_fn(denoised_pred, denoised_truth).mean() * loss_mults[i_nr]
            
            #display results of training
            if it % 100 == 0:
                with torch.no_grad():
                    i=0
                    plt_imshow_pt(torch.cat([noisy_image[i],denoised_pred[i],denoised_truth[i]],2)*255)
                    plt.show()
        #unblur task
        if loss_restore_mult > 0:
            loss += loss_fn(model(damage_f(batch['image'])), batch['image']).mean() * loss_restore_mult
        #preserve task
        if loss_preserve_mult > 0:
            loss += loss_fn(model(batch['image']), batch['image']).mean() * loss_preserve_mult
        
    #compute gradients
    scaler.scale(loss).backward()
    
    #apply gradients
    scaler.step(optimizer)
    
    #zero gradients
    scaler.update()
    optimizer.zero_grad()
    
            
    #val
    #measure
    if it % 100 == 0:
        with torch.no_grad():
            pred_classes_raw = Counter(imagenet_classifier(FT.resize(normalize_t(batch['image']), 224).cpu()).argmax(1).tolist())
            noise_images = torch.rand([len(batch['image']),3,img_size, img_size]).cuda(non_blocking=True)
            for i in range(len(noise_rates)+1):
                noise_images = model(noise_images)
            pred_classes_gen = Counter(imagenet_classifier(FT.resize(normalize_t(noise_images), 224).cpu()).argmax(1).tolist())
            prop_found = {cl:min(1,pred_classes_gen.get(cl,0)/pred_classes_raw[cl]) for cl in pred_classes_raw}
            val_score = sum([prop_found[cl]*pred_classes_raw[cl] for cl in pred_classes_raw]) / len(batch['image'])
            val_scores[it] = val_score
            time_to_valscore[time.time()-start_loop_time] = val_score
    #show
    if it % 100 == 0 and it > start_iter or it in [0, 5, 20, 50, 80]:
        with torch.no_grad():
            noisy_image = torch.rand([32,3,img_size, img_size]).cuda(non_blocking=True)
            plt_imshow_pt(noisy_image[0]*255)
            plt.savefig(join(get_log_dir(),f'{it}_0.png'))
            for i in range(1,len(noise_rates)+3): #+50
                noisy_image = model(noisy_image)
                plt_imshow_pt(noisy_image[0]*255)
                plt.savefig(join(get_log_dir(),f'{it}_{i}.png'))
            plt.show()
        
    
    #log
    loss = loss.item()
    delta_t = round(time.time() - last_it_time, 2)
    last_it_time = time.time()
    print(f'iteration: {it} | loss: {loss:.2f} | Δt: {delta_t} s')
    train_losses[it] = loss
    
    #plot 
    if it % 50 == 0 and it > 0:
        train_losses_blur = gaussian_filter(list(train_losses.values()), 5)
        train_losses = {i:v for i,v in enumerate(train_losses_blur)}
        plot_dict(train_losses, l='train')
        plt.show()
        plot_dict(time_to_valscore, l='val', color='orange')
        plt.ylim(0,1)
        plt.savefig(join(get_log_dir(),'last_val_score.png'))
        plt.show()
    
    #delay
    cur_time = time.time() - start_loop_time
    if cur_time > max_runtime:
        break
    
    # #save
    if it % 3000 == 0 and it > start_iter:
        #save
        chkpt_pth = join(get_log_dir(),f'denoiser_{it}.ckpt')
        torch.save(model.state_dict(), chkpt_pth)
        



#build grid results
for n_pass in [1, 2, 3, 4, 5, 10, 50, 100]:
    denoised_image = torch.rand([256,3,img_size, img_size]).cuda(non_blocking=True)
    with torch.no_grad():
        for i in range(1,len(noise_rates)+n_pass): #+50
            denoised_image = model(denoised_image)
    #put it on the grid
    w,h=10,5
    grid = torch.zeros((3, img_size*h, img_size*w))
    i_w,i_h = 0,0
    for i in range(len(denoised_image)):
        grid[:, (i_h*img_size):(i_h+1)*img_size,(i_w*img_size):(i_w+1)*img_size] = denoised_image[i]
        i_w += 1
        if i_w == w:
            i_w = 0
            i_h += 1
        if i_h == h:
            break
    #show/save
    plt.figure(figsize=(10,5))
    plt_imshow_pt(grid*255)
    plt.savefig(join(get_log_dir(),f'{it}_end_result_{n_pass}pass.png'))
    plt.show()
    


















