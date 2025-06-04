# T2I-shortcut
Image synthesis via shortcut model and CLIP layer


This project is part of the "Generative AI" course in BGU. 
It relies on the paper "One Step Diffusion via Shortcut Models" (https://arxiv.org/pdf/2410.12557) and its code presented in https://github.com/kvfrans/shortcut-models

This version based on PyTorch instead of JAX library to utilize GPUs. Special thanks to smileyenot983 for the help with the PyTorch implementation.

Attached to the repo also the project summary and presentation.

## Dataset
celeba-hq dataset from HuggingFace https://huggingface.co/datasets/mattymchen/celeba-hq for training.

## Training samples

![1, 2, 4 steps](https://github.com/anastasiaIvanovski/T2I-shortcut/blob/main/Training%20samples/training_samples.png?raw=true)

![128 steps](https://github.com/anastasiaIvanovski/T2I-shortcut/blob/main/Training%20samples/training_samples_128.png?raw=true)

