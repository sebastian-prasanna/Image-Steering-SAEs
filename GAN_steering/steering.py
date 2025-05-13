import torch
import einops
import tqdm

# Steer the 'average image' to see what a steering vector does

@torch.inference_mode()
def activations2image(activations, SAE, G):
    ''' 
    Activations of shape (b, 1024)
    '''
    with torch.no_grad():
        latent = SAE.decode(activations)
    latent = einops.repeat(latent, 'b d -> b n d', n = 14)
    with torch.no_grad():
        images = G.synthesis(latent)
    return images

def steer_avg_activations(G, SAE, avg_activations, idx, device = 'mps', min = -3, max = 3, steps = 7):
    steers = torch.zeros((steps, 1024)).to(device)
    steers[:, idx] += torch.linspace(start = min, end = max, steps = steps).to(device)
    activations = avg_activations + steers
    images = activations2image(activations, SAE, G)
    return images


@torch.inference_mode()
def steer(G, SAE, test_loader, idx, device = 'mps', num_imgs = 1, min = -4, max = 4, steps = 9):
    ''' 
    idx gives the index of the steering vector
    num_imgs should be less than the inside batch dimension of the test_loader (in this case 64)
    Unfortunately, if num_imgs is large, we run into memory problems, so we run a for loop.
    ''' 
    coefficients = torch.linspace(start = min, end = max, steps = steps).to(device)
    steering_vector = SAE.encoder.weight[idx, :]
    # steers has shape (steps, 512)
    steers = coefficients[:, None] * steering_vector
    batch = next(iter(test_loader)).to(device)
    # latents has shape (num_imgs, 512)
    latents = batch[0, :num_imgs, :]
    # reconstructions has shape (num_images, 512)
    reconstructions, sparse_representations = SAE(latents)
    final_latents = reconstructions[:, None, :] + steers[None, :, :]
    # final_latents has shape (num_imgs * steps, 14, 512)
    final_latents = einops.repeat(final_latents, 'num_imgs steps latent_dim -> num_imgs steps num_layers latent_dim', num_layers = 14)
    images = []
    for i in range(num_imgs):
        steered_image = G.synthesis(final_latents[i])
        images.append(steered_image)
    images = torch.cat(images, dim = 0)
    return images

@torch.inference_mode()
def get_top_latents(SAE, dataset, idx, device = 'mps', k = 16, largest = True):
    activations = torch.tensor([]).to(device)
    for i in tqdm.tqdm(range(len(dataset))):
        batch = dataset[i]
        reconstructions, sparse_representations = SAE(batch)
        batch_activations = sparse_representations[:, idx]
        activations = torch.cat((activations, batch_activations), dim = 0)
    topk, indices = torch.topk(activations, k = k, largest = largest)
    print(topk)
    batch_indices = indices // 64
    inside_batch_indices = indices % 64
    latents = []
    for i in range(len(batch_indices)):
        batch_index = batch_indices[i].item()
        inside_batch_index = inside_batch_indices[i].item()
        latents.append(dataset[batch_index][inside_batch_index])
    latents = torch.stack(latents, dim = 0)
    latents = einops.repeat(latents, 'b d -> b n d', n = 14)
    return latents