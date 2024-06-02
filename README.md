# ACD
Codes of paper "Unified Conditional Image Generation for Visible-Infrared Person Re-Identification" (under review).

 <img src="/figs/motivation.png" width="560" height="420">
 
## Motivation of our method.
As shown in the above figure, given pedestrian contours extracted by high frequency filtering, we are able to produce diverse and semantically aligned intra-modality, middle-modality, and cross-modality images. Specifically, we adapt the conditional diffusion model to generate desired images from random Gaussian noises, whose generative process is conditioned on the modality information and modality-irrelative pedestrian contours.

 <img src="/figs/framework.png" width="1000" height="320">
 
## Framework of our method.
This figure takes the visible and infrared image generation as example, which enables intra-modality and cross-modality image generation. While the middle-modality image generation is not presented in this figure. The
forward process gradually adds random noise on true images without learnable parameters, which is the same as existing unconditional diffusion models. The reverse process contains the conditional denoising and modal adversarial training. 

## Process.
1. Run main_fft.py to generate pedestrian contours;
2. Run train_model_cross_and_intra.sh for cross- and intra-modal image generation;
3. Run train_model_middle.sh for middle-modal image generation.
