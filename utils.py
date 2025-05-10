import matplotlib.pyplot as plt
import torch
import einops

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
    for i in range(h):
        for j in range(w):
            to_show = reshaped[h * i + j]
            ax[i, j].imshow(to_show)
            ax[i, j].axis('off')

def process(imgs):
    # Maps pixel values from [-1, 1] to [0, 1]
    return torch.clamp((imgs + 1) * 0.5, 0, 1)