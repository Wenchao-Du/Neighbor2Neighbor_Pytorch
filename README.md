# Neighbor2Neighbor_Pytorch
This is an unofficial implementation for the paper "NEIGHBOR2NEIGHBOR: SELF-SUPERVISED DENOISING FROM SINGLE NOISY IMAGES"
# Introduction
This is an implementation for the paper "NEIGHBOR2NEIGHBOR"<https://arxiv.org/abs/2101.02824>, and i have tied this method for real noise removal task in my own dataset, which has presented some effects to some degree. 
But i haven't applied it to some traditional denoising tasks, e.g., Gaussian nosie and Posson noise removing. You could try it easily in this codes.

# Basic requirements
1. Pytorch > 1.3.0
2. Nvidia apex ( this codes are easily used for multi-gpus training)
3. Some Python packages

# Training and testing
1. First, you need to add the data preprocessing file based your tasks.
2. Update the config.yaml file
3. training and testing with `python main.py`

# Extra discussions
This is an extended work for *Noise2Noise* <https://arxiv.org/abs/1803.04189>, which all depend on the powerful zero-mean noise prior hypothesis. 
