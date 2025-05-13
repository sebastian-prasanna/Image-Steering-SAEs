import torch
from utils import display_img, process
import einops

def test_reconstructions(G, sae, loader, h = 4, w = 4, display = True):
    '''
    h * w should be less than 64, or else the code won't work. It's probably too small to see anyway for h, w > 8
    '''
    batch = next(iter(loader))
    # batch is batch of batches, so it's gonna be something like (batch_dim1, batch_dim2, latent_dim_size)
    # usually batch_dim1 = batch_dim2 = 64
    total_imgs = h * w
    latents = batch[0, :total_imgs, :]
    with torch.no_grad():
        reconstructed_latents, sparse_representation = sae(latents)
    imgs = process(G(latents, None))
    reconstructed_imgs = process(G(reconstructed_latents, None))
    if display:
        print('Original Images:')
        display_img(imgs, h, w)
        print('Reconstructed Images')
        display_img(reconstructed_imgs, h, w)
    return imgs, reconstructed_imgs

def get_top_k_images(G, sae, num_images = 128, k = 16, device = 'mps'):
    ''' 
    Take a trained SAE and generate images using random noise
    '''
    noise = torch.randn((num_images, G.z_dim), device = device)
    c = None
    with torch.no_grad():
        w = G.mapping(noise, c)
        # to take out the middle dimension of 14
        w = w[:, 0, :]
        reconstructions, sparse_representations = sae(w)
        w = einops.repeat(w, 'b d -> b l d', l = 14)
        images = G.synthesis(w)
        # images should be a tensor of shape (batch_size, 3, 256, 256)
        # sparse_representations should be a tensor of shape 
