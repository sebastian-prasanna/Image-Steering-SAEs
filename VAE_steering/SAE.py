import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, sparsity_lambda=1e-3, device = 'mps'):
        super().__init__()
        self.encoder = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.relu = torch.nn.ReLU()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.sparsity_lambda = sparsity_lambda
        self.device = device

    def decode(self, z):
        # decoder with tied weights
        return F.linear(z, self.encoder.weight.T)

    def forward(self, x):
        z = self.encoder(x)
        z = self.relu(z)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, x, x_hat, z):
        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = z.abs().mean()
        return recon_loss + self.sparsity_lambda * sparsity_loss, recon_loss, sparsity_loss

    # periodically do this in training loop
    def resample_dead_features(self, data_batch, z_batch, activation_threshold=1e-4):
        """
        Resample dead features in the encoder weight matrix.

        data_batch: [B1, B2, latent_dim] — input latent vectors
        z_batch:    [B1, B2, hidden_dim] — encoded outputs
        """
        # Flatten batch dimensions: shape → [B1 * B2, latent_dim or hidden_dim]
        flat_data = data_batch.view(-1, data_batch.size(-1))       # shape: [B, latent_dim]
        flat_z = z_batch.view(-1, z_batch.size(-1))                 # shape: [B, hidden_dim]

        # Mean absolute activation per neuron
        mean_activation = flat_z.abs().mean(dim=0)  # shape: [hidden_dim]

        # Indices of features that are mostly inactive
        dead_features = (mean_activation < activation_threshold).nonzero(as_tuple=True)[0]

        if len(dead_features) == 0:
            print('None Resampled!')
            return None

        # Sample random data points from the batch
        num_dead = len(dead_features)
        print(f'{num_dead} resampled!')
        rand_indices = torch.randint(0, flat_data.size(0), (num_dead,))
        sampled_inputs = flat_data[rand_indices]  # shape: [num_dead, latent_dim]

        # Normalize to unit norm
        sampled_inputs = F.normalize(sampled_inputs, p=2, dim=-1)

        # Replace encoder weights
        with torch.no_grad():
            self.encoder.weight[dead_features] = sampled_inputs