# Denoising to generate images
Custom implementation of the denoising task to generate images. This repo is an experiment.  
Modern (2020-2023) models used for image generation are based on the denoising task, I tried to re-implement this idea from scratch.  
This repo may not follow the usual formalism for this task, it was coded based on what was the most intuitive to me.  
The goal of this repo isn't to be competitive with other image generation algorithms, results I had and experiments I did are very far from SOTA.

# Results

Results after 17500 training steps (~2h on middle-end GPU):  
- Video:  
https://user-images.githubusercontent.com/12411288/228590068-5b4f1c60-f5bf-4a42-b237-8da288428184.mp4

- Grid:  
![17511_end_result_10pass](https://user-images.githubusercontent.com/12411288/228590451-94f4ee19-3c79-438a-b686-bb1aa8ff321d.png)








