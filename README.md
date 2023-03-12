# DiffusionIQT
Image Quality Transfer in Medical Imaging using Diffusion Models

2023.02 -> Test on timesteps, T is reduced from 1000 to 500. The inference time reduced to 25s from 49s. Very slight increase in l1 distance, but no visual change in the results

2023.03 -> Test on weights, Current weighting scheme (high weight for high SNR) blurrs the output, no weighting is the best so far, inverse is okay

2023.03.13 -> 3D conversion
