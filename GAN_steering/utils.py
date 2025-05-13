import matplotlib.pyplot as plt
import torch
import einops
import numpy as np

def display_img(img, h, w):
    ''' 
    Takes an image of dimension (batch_dim, channels, height, width), and prints them in a h x w grid.
    '''
    if img.device.type != 'cpu':
        img = img.cpu()
    reshaped = einops.rearrange(img, 'b c h w -> b h w c').cpu().numpy()
    if h == 1 and w == 1:
        plt.imshow(reshaped[0])
        plt.show()
        return None
    fig, ax = plt.subplots(h, w, figsize=(10, 10))
    # Ensure ax is always 2D for consistent indexing
    if h == 1:
        ax = np.expand_dims(ax, axis=0)
    if w == 1:
        ax = np.expand_dims(ax, axis=1)
    for row in range(h):
        for col in range(w):
            to_show = reshaped[w * row + col]
            ax[row, col].imshow(to_show)
            ax[row, col].axis('off')
    plt.tight_layout()
    plt.show()

def process(imgs):
    # Maps pixel values from [-1, 1] to [0, 1]
    return torch.clamp((imgs + 1) * 0.5, 0, 1)

@torch.inference_mode()
def sample(GAN, latents):
    ''' 
    Samples from the GAN given latents of shape (Batch_size, 512)
    Processes the images to the range [0, 1] for displaying
    '''
    images = GAN.synthesis(latents)
    images = process(images)
    return images