# Denoising to generate images
Custom implementation of the denoising task to generate images. This repo is an experiment.  
Modern (2020-2023) models used for image generation are based on the denoising task, I tried to re-implement this idea from scratch. 
This repo may not follow the usual formalism for this task, it was coded based on what was the most intuitive to me.  
The goal of this repo isn't to be competitive with other image generation algorithms, results I had and experiments I did are very far from SOTA.

# Results

Results after 17500 training steps (~2h on middle-end GPU):  
- Video:  

  https://user-images.githubusercontent.com/12411288/228590866-873a5a5d-befc-4f0f-b5b4-31e3386e7451.mp4



- Grid:  
![17511_end_result_10pass](https://user-images.githubusercontent.com/12411288/228590451-94f4ee19-3c79-438a-b686-bb1aa8ff321d.png)

# Dataset

dogs.pickle contains 3602 low-res (64*64) images of dogs.

# Requirements
- spyder / (interactive python session): https://github.com/spyder-ide/spyder or disable all plots
- lpips: https://github.com/richzhang/PerceptualSimilarity 
- timm: https://github.com/huggingface/pytorch-image-models
- lion: https://github.com/google/automl/tree/master/lion (included)
- eca: https://github.com/BangguWu/ECANet (included)
- a GPU (8GB of VRAM recommended)







