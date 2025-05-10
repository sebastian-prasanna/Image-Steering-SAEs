import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Convolutional encoder that maps an image to latent mean and log-variance vectors.
    We're going to be workign with 64 by 64 images
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super(Encoder, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)  # -> 32 x 32 x 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)           # -> 64 x 16 x 16
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)           # -> 64 x 8 x 8
        # Flatten and linear layers for mean and logvar
        self.flatten_dim = 64 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.flatten_dim)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """
    Convolutional decoder that maps latent vectors back to images.
    """
    def __init__(self, out_channels: int = 3, latent_dim: int = 128):
        super(Decoder, self).__init__()
        # Linear layer to expand latent vector
        self.flatten_dim = 64 * 8 * 8
        self.fc = nn.Linear(latent_dim, self.flatten_dim) # -> 64 x 8 x 8
        # Transposed convolutions for upsampling
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size = 4, stride = 2, padding = 1) # -> 64 x 16 x 16
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # -> 32 x 32 x 32
        self.deconv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)  # -> out_channels x 64 x 64

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc(z))
        x = x.view(-1, 64, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # output in [0,1]
        return x


class VAE(nn.Module):
    """
    Variational Autoencoder combining the Encoder and Decoder.
    """
    def __init__(self, device = 'mps', in_channels: int = 3, latent_dim: int = 128):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)
        self.latent_dim = latent_dim
        self.device = device

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Reparameterization trick: z = mu + sigma * eps
        # 0.5 because we need to take square root of variance
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        BCE = F.mse_loss(recon_x, x, reduction='mean')
        # KL divergence
        beta = 0.1
        KLD = (-0.5) * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD, BCE, KLD
    
def sample(self, num_samples, latent_dim):
    latents = torch.rand(size = (num_samples, latent_dim))
    latents = latents.to(self.device)
    with torch.no_grad():
        generated_images = self.decoder(latents)
    return generated_images

def reconstruct(self, batch):
    batch = batch.to(self.device)
    with torch.no_grad():
        reconstructions = self.forward(batch)
    return reconstructions