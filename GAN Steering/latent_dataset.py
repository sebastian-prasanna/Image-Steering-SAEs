import torch
from torch.utils.data import Dataset
import os
import tqdm

class WPlusLatentsDataset(Dataset):
    def __init__(self, latent_dir):
        """
        Args:
            latent_dir (str): Path to the directory containing .pt latent files
        """
        self.latent_dir = latent_dir

    def __len__(self):
        return len([f for f in os.listdir(self.latent_dir) if os.path.isfile(os.path.join(self.latent_dir, f))])

    def __getitem__(self, idx):
        file_path = os.path.join(self.latent_dir, str(idx) + '.pt')
        latent = torch.load(file_path)  # should be shape [64, 14, 512]
        return latent
    

def generate_latents(G, device, batch_size, use_w_plus):
    z = torch.randn([batch_size, G.z_dim], device = device)
    c = None
    with torch.no_grad():
        w = G.mapping(z, c)
    # w is of dimension (batch_size, 14, 512)
    if use_w_plus:
        return w
    else:
        return w[:, 0, :]
    
def save_latents(G, folder_path, num_batches, device = 'mps', batch_size = 64, use_w_plus = False):
    os.makedirs(folder_path, exist_ok = True)
    # folder_path = '/Users/spra/Desktop/Personal Projects/Image-Steering-SAEs/G_w_latents/'
    for i in tqdm.tqdm(range(num_batches)):
        latents = generate_latents(G, device = device, batch_size = batch_size, use_w_plus = use_w_plus)
        path = os.path.join(folder_path, str(i) + '.pt')
        torch.save(latents, path)