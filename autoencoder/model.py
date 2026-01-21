import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=4, padding=0),  # 224 -> 56
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=4, stride=4, padding=0),  # 56 -> 14
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.fc_enc = nn.Linear(14 * 14 * 32, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 14 * 14 * 32)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4),  # 14 -> 56
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=4),  # 56 -> 224
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc_enc(h)

        h = self.fc_dec(z)
        h = h.view(h.size(0), 32, 14, 14)
        out = self.decoder(h)
        return out, z
