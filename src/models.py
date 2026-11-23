import torch
import torch.nn as nn

from .config import ORIGINAL_DIM, HIDDEN_DIM, LATENT_DIM

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(ORIGINAL_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, ORIGINAL_DIM),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.flatten(1)
        z = self.encoder(x)
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(ORIGINAL_DIM, HIDDEN_DIM)
        self.mean = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.log_var = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, ORIGINAL_DIM)
        )

    def encode(self, x):
        h = torch.relu(self.hidden(x.flatten(1)))
        return self.mean(h), self.log_var(h)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        noise = torch.randn_like(std)
        return mean + std * noise

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z), mean, log_var


class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(noise_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Tanh()
        )

    def forward(self, z):
        return self.network(z).reshape(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Dropout(0.3), nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.network(x)


class DCGenerator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 128, 7, 1, 0, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False), nn.Tanh()
        )

    def forward(self, z):
        return self.network(z.reshape(z.size(0), -1, 1, 1))


class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 7, 1, 0, bias=False), nn.Flatten()
        )

    def forward(self, x):
        return self.network(x)


def initialize_dcgan(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight, 0.0, 0.02)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.zeros_(layer.bias)


class TimeEmbedding(nn.Module):
    def __init__(self, dimension=64):
        super().__init__()
        self.dimension = dimension
        self.network = nn.Sequential(nn.Linear(dimension, dimension), nn.SiLU(), nn.Linear(dimension, dimension))

    def forward(self, t):
        half = self.dimension // 2
        frequencies = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device)) *
            torch.arange(half, device=t.device) / (half - 1)
        )
        angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return self.network(torch.cat([angles.sin(), angles.cos()], dim=1))


class DenoisingBlock(nn.Module):
    def __init__(self, input_channels, output_channels, time_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, output_channels)
        self.norm2 = nn.GroupNorm(8, output_channels)
        self.time_layer = nn.Linear(time_dim, output_channels)
        self.shortcut = nn.Conv2d(input_channels, output_channels, 1)
        self.activation = nn.SiLU()

    def forward(self, x, time_embedding):
        h = self.activation(self.norm1(self.conv1(x)))
        h = h + self.time_layer(time_embedding)[:, :, None, None]
        h = self.activation(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding()
        self.block1 = DenoisingBlock(1, 32)
        self.down1 = nn.Conv2d(32, 64, 4, 2, 1)
        self.block2 = DenoisingBlock(64, 64)
        self.down2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.middle = DenoisingBlock(128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.block3 = DenoisingBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.block4 = DenoisingBlock(64, 32)
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        embedding = self.time_embedding(t)
        skip1 = self.block1(x, embedding)
        skip2 = self.block2(self.down1(skip1), embedding)
        h = self.middle(self.down2(skip2), embedding)
        h = self.block3(torch.cat([self.up1(h), skip2], dim=1), embedding)
        h = self.block4(torch.cat([self.up2(h), skip1], dim=1), embedding)
        return self.output(h)


class MLPClassifier(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 10)

    def encode(self, x):
        h = torch.relu(self.fc1(x.flatten(1)))
        return torch.relu(self.fc2(h))

    def forward(self, x):
        return self.fc3(self.encode(x))


class FrozenDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 784), nn.Sigmoid()
        )

    def forward(self, z):
        return self.network(z)
