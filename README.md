# Conditional VAE for Chest X-rays

Our implementation trains a convolutional conditional variational autoencoder
on the NORMAL and PNEUMONIA images. It uses a class-balanced training sampler,
a stratified validation split, spatial and feature-wise class conditioning, KL
warm-up, and an L1 + SSIM + edge reconstruction objective.

