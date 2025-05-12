import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
from vae_stuff.VAE import VAE
from SAE import SparseAutoencoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def load_models(vae_path, sae_path, device):
    """Load trained VAE and SAE models"""
    vae = VAE(device=device, latent_dim=128).to(device)
    sae = SparseAutoencoder(latent_dim=128, hidden_dim=512, device=device).to(device)
    
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    
    vae.eval()
    sae.eval()
    return vae, sae

def get_test_images(dataloader, num_samples=5):
    """Get a batch of test images"""
    test_batch, _ = next(iter(dataloader))
    return test_batch[:num_samples]

def analyze_feature(vae, sae, images, feature_idx, scales=[-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda'):
    """Analyze a single feature by creating variations"""
    with torch.no_grad():
        # Get VAE latents
        mu, _ = vae.encoder(images.to(device))
        
        # Get SAE activations
        z = sae.encoder(mu)
        
        # Get feature vector
        feature_vector = sae.encoder.weight[feature_idx]
        
        # Create variations
        variations = []
        for scale in scales:
            # Add scaled feature vector to VAE latents
            modified_mu = mu + scale * feature_vector
            # Decode back to images
            recon = vae.decoder(modified_mu)
            variations.append(recon)
        
        # Get feature activation statistics
        activation = z[:, feature_idx].abs().mean().item()
        
        return torch.cat(variations, dim=0), activation

def visualize_feature_effect(vae, sae, images, feature_idx, output_dir='feature_analysis', device='cuda'):
    """Create detailed visualization of a feature's effect"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get variations
    variations, activation = analyze_feature(vae, sae, images, feature_idx, device=device)
    
    # Create grid of images
    grid = make_grid(variations, nrow=len(images), padding=2, normalize=True)
    
    # Save visualization
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.title(f'Feature {feature_idx} (Mean Activation: {activation:.4f})')
    plt.axis('off')
    plt.savefig(f'{output_dir}/feature_{feature_idx}_analysis.png')
    plt.close()
    
    return activation

def find_semantic_features(vae, sae, dataloader, num_features=20, device='cuda'):
    """Analyze multiple features and identify potentially semantic ones"""
    test_images = get_test_images(dataloader)
    
    # Analyze features
    activations = []
    for feature_idx in range(num_features):
        activation = visualize_feature_effect(vae, sae, test_images, feature_idx, device=device)
        activations.append((feature_idx, activation))
    
    # Sort features by activation
    activations.sort(key=lambda x: x[1], reverse=True)
    
    print("\nFeature Analysis Results:")
    print("------------------------")
    print("Top activated features:")
    for idx, activation in activations[:5]:
        print(f"Feature {idx}: Mean activation = {activation:.4f}")
    
    print("\nLeast activated features:")
    for idx, activation in activations[-5:]:
        print(f"Feature {idx}: Mean activation = {activation:.4f}")
    
    return activations

def manipulate_image(vae, sae, image, feature_idx, scale=1.0, device='cuda'):
    """Manipulate a single image using a specific feature"""
    with torch.no_grad():
        # Get VAE latent
        mu, _ = vae.encoder(image.unsqueeze(0).to(device))
        
        # Get feature vector
        feature_vector = sae.encoder.weight[feature_idx]
        
        # Apply feature manipulation
        modified_mu = mu + scale * feature_vector
        
        # Decode back to image
        modified_image = vae.decoder(modified_mu)
        
        return modified_image.squeeze(0)

def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Load models (update these paths to your saved models)
    vae_path = 'results/vae_epoch_15.pt'  # or whatever the latest VAE checkpoint is
    sae_path = 'results/sae_epoch_25.pt'  # or whatever the latest SAE checkpoint is
    vae, sae = load_models(vae_path, sae_path, device)
    
    # Setup dataloader
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root='data/celeba', transform=transform)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    # Analyze features
    print("Analyzing features...")
    activations = find_semantic_features(vae, sae, dataloader, num_features=20, device=device)
    
    # Interactive feature manipulation
    print("\nTo manipulate images with specific features:")
    print("1. Look at the generated visualizations in the 'feature_analysis' directory")
    print("2. Identify features that seem to control interesting attributes")
    print("3. Use the manipulate_image() function to apply those features")
    print("\nExample usage:")
    print("modified_image = manipulate_image(vae, sae, image, feature_idx=5, scale=2.0)")

if __name__ == "__main__":
    main() 