import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from tqdm import tqdm
import numpy as np
from vae_stuff.VAE import VAE
from SAE import SparseAutoencoder
import matplotlib.pyplot as plt

# Constants
BATCH_SIZE = 128
VAE_LATENT_DIM = 128
SAE_HIDDEN_DIM = 512
NUM_EPOCHS_VAE = 15
NUM_EPOCHS_SAE = 25
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
CELEBA_PATH = 'data/celeba'
LATENTS_PATH = 'data/latents.pt'

def get_celeba_dataloader():
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(root=CELEBA_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    return dataloader

def train_vae(vae, dataloader, optimizer, epoch):
    vae.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc=f'VAE Epoch {epoch}')):
        data = data.to(DEVICE)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = vae(data)
        loss, bce, kld = vae.loss_function(recon_batch, data, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'VAE Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f} BCE: {bce:.4f} KLD: {kld:.4f}')
            
    return total_loss / len(dataloader)

def extract_latents(vae, dataloader, num_samples=10000):
    """Extract and store latent representations from VAE"""
    vae.eval()
    latents = []
    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc='Extracting latents'):
            data = data.to(DEVICE)
            mu, _ = vae.encoder(data)
            latents.append(mu.cpu())
            
            if len(latents) * BATCH_SIZE >= num_samples:
                break
    
    latents = torch.cat(latents, dim=0)[:num_samples]
    torch.save(latents, LATENTS_PATH)
    print(f"Saved {len(latents)} latent vectors to {LATENTS_PATH}")
    return latents

def train_sae(sae, latents, optimizer, epoch):
    sae.train()
    total_loss = 0
    total_recon = 0
    total_sparsity = 0
    
    # Create batches from latents
    indices = torch.randperm(len(latents))
    for i in range(0, len(latents), BATCH_SIZE):
        batch_indices = indices[i:i + BATCH_SIZE]
        batch = latents[batch_indices].to(DEVICE)
        
        optimizer.zero_grad()
        x_hat, z = sae(batch)
        loss, recon_loss, sparsity_loss = sae.loss(batch, x_hat, z)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_sparsity += sparsity_loss.item()
        
        # Resample dead features periodically
        if i % (BATCH_SIZE * 10) == 0:
            sae.resample_dead_features(batch, z)
    
    avg_loss = total_loss / (len(latents) / BATCH_SIZE)
    avg_recon = total_recon / (len(latents) / BATCH_SIZE)
    avg_sparsity = total_sparsity / (len(latents) / BATCH_SIZE)
    
    print(f'SAE Epoch {epoch} Loss: {avg_loss:.4f} Recon: {avg_recon:.4f} Sparsity: {avg_sparsity:.4f}')
    return avg_loss

def analyze_features(vae, sae, test_dataloader, num_samples=5):
    """Analyze SAE features by manipulating them in the VAE latent space"""
    vae.eval()
    sae.eval()
    
    # Get a batch of test images
    test_batch, _ = next(iter(test_dataloader))
    test_batch = test_batch[:num_samples].to(DEVICE)
    
    # Get VAE latents
    with torch.no_grad():
        mu, _ = vae.encoder(test_batch)
        z = sae.encoder(mu)
        
        # Analyze each feature
        for feature_idx in range(min(10, sae.hidden_dim)):  # Look at first 10 features
            # Get the feature vector
            feature_vector = sae.encoder.weight[feature_idx]
            
            # Create variations by adding/subtracting the feature
            variations = []
            for scale in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                modified_mu = mu + scale * feature_vector
                recon = vae.decoder(modified_mu)
                variations.append(recon)
            
            # Save the variations
            variations = torch.cat(variations, dim=0)
            save_image(variations, f'feature_{feature_idx}_variations.png', nrow=num_samples)
            
            # Print feature activation statistics
            activation = z[:, feature_idx].abs().mean().item()
            print(f"Feature {feature_idx} mean activation: {activation:.4f}")

def main():
    # Create directories
    os.makedirs(CELEBA_PATH, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Initialize models
    vae = VAE(device=DEVICE, latent_dim=VAE_LATENT_DIM).to(DEVICE)
    sae = SparseAutoencoder(latent_dim=VAE_LATENT_DIM, hidden_dim=SAE_HIDDEN_DIM, device=DEVICE).to(DEVICE)
    
    # Get dataloader
    dataloader = get_celeba_dataloader()
    
    # Train VAE
    print("Training VAE (15 epochs)...")
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
    for epoch in range(NUM_EPOCHS_VAE):
        train_vae(vae, dataloader, vae_optimizer, epoch)
        if (epoch + 1) % 5 == 0:
            torch.save(vae.state_dict(), f'results/vae_epoch_{epoch+1}.pt')
    
    # Extract latents
    print("Extracting latent representations...")
    latents = extract_latents(vae, dataloader)
    
    # Train SAE
    print("Training SAE (25 epochs)...")
    sae_optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)
    for epoch in range(NUM_EPOCHS_SAE):
        train_sae(sae, latents, sae_optimizer, epoch)
        if (epoch + 1) % 5 == 0:
            torch.save(sae.state_dict(), f'results/sae_epoch_{epoch+1}.pt')
    
    # Analyze features
    print("Analyzing features...")
    analyze_features(vae, sae, dataloader)

if __name__ == "__main__":
    main() 