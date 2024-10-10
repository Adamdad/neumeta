import torch
import torch.nn as nn
import torch.nn.functional as F

class CelebAVAE(nn.Module):
    def __init__(self, init_channels=64, image_channels=3, latent_dim=100, h_var=None):
        super(CelebAVAE, self).__init__()
        if h_var is None:
            self.h_var = init_channels
        else:
            self.h_var = h_var
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, init_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels),
            nn.LeakyReLU(0.2),
            
            
            nn.Conv2d(init_channels, init_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels * 2),
            nn.LeakyReLU(0.2),
            
            
            nn.Conv2d(init_channels * 2, init_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels * 4),
            nn.LeakyReLU(0.2),
            
            
            nn.Conv2d(init_channels * 4, init_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels * 8),
            nn.LeakyReLU(0.2),

            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc_mu = nn.Linear(init_channels * 8 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(init_channels * 8 * 4 * 4, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, init_channels * 8 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(init_channels * 8, init_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels * 4),
            nn.LeakyReLU(0.2),
            
            
            nn.ConvTranspose2d(init_channels * 4, self.h_var * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.h_var * 2),
            nn.LeakyReLU(0.2),
            
            
            nn.ConvTranspose2d(self.h_var * 2, self.h_var, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.h_var),
            nn.LeakyReLU(0.2),
            
            
            nn.ConvTranspose2d(self.h_var, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        encoded = encoded.view(batch_size, -1)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        return mu, log_var

    def decode(self, z):
        expanded = self.decoder_input(z).view(z.size(0), -1, 4, 4)
        return self.decoder(expanded)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var
    
    @property
    def learnable_parameter(self):
        self.keys = [k for k, w in self.named_parameters() if k.startswith(f'decoder.3') or k.startswith(f'decoder.6') or k.startswith(f'decoder.9')]
        return {k:w for k, w in self.state_dict().items() if k in self.keys}


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, h_var=None):
        super(VAE, self).__init__()
        self.h_var = h_var if h_var is not None else h_dim1
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, self.h_var)
        self.fc6 = nn.Linear(self.h_var, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    @property
    def learnable_parameter(self):
        self.keys = [k for k, w in self.named_parameters(
        ) if k.startswith(f'fc5') or k.startswith(f'fc6')]
        return {k: v for k, v in self.state_dict().items() if k in self.keys}
