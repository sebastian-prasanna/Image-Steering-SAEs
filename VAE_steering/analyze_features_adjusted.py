import torch
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from VAE import VAE
from SAE import SparseAutoencoder
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load models
vae_path = 'results/vae_epoch_15.pt'
sae_path = 'results/sae_epoch_25.pt'
vae = VAE(device=DEVICE, latent_dim=128).to(DEVICE)
sae = SparseAutoencoder(latent_dim=128, hidden_dim=512, device=DEVICE).to(DEVICE)
vae.load_state_dict(torch.load(vae_path, map_location=DEVICE))
sae.load_state_dict(torch.load(sae_path, map_location=DEVICE))
vae.eval()
sae.eval()

def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root='data/celeba', transform=transform)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    return dataloader

def analyze_features(vae, sae, test_dataloader, num_samples=5):
    """Analyze the most active SAE features by manipulating them in the VAE latent space"""
    vae.eval()
    sae.eval()
    
    # Get a batch of test images
    test_batch, _ = next(iter(test_dataloader))
    test_batch = test_batch[:num_samples].to(DEVICE)
    
    # Get VAE latents
    with torch.no_grad():
        mu, _ = vae.encoder(test_batch)
        z = sae.encoder(mu)  # shape: [batch, hidden_dim]
        
        # Compute mean activation for each feature
        activations = z.abs().mean(dim=0)  # shape: [hidden_dim]
        # Get indices of top 10 most active features
        topk = torch.topk(activations, 10).indices.tolist()
        
        print("Top 10 most active features:", topk)
        
        for feature_idx in topk:
            feature_vector = sae.encoder.weight[feature_idx]
            variations = []
            for scale in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                modified_mu = mu + scale * feature_vector
                recon = vae.decoder(modified_mu)
                variations.append(recon)
            variations = torch.cat(variations, dim=0)
            save_image(variations, f'feature_{feature_idx}_variations_top_active.png', nrow=num_samples)
            activation = z[:, feature_idx].abs().mean().item()
            print(f"Feature {feature_idx} mean activation: {activation:.4f}")

def main():
    dataloader = get_dataloader()
    analyze_features(vae, sae, dataloader, num_samples=5)

if __name__ == "__main__":
    main() 