import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from IPython.display import clear_output

from config import device

class ConvolutionalVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvolutionalVAE, self).__init__()
        # self.loss_dict = {
        #     'reconstruction_loss': [],
        #     'kl_divergence': []}
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), # batch_size, 16, 64, 64
            nn.MaxPool2d(2, stride=2), # batch_size, 16, 32, 32
            nn.BatchNorm2d(16),
            nn.LeakyReLU(), 
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # batch_size, 32, 16, 16
            nn.MaxPool2d(2, stride=2), # batch_size, 8, 8, 8
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # batch_size, 64, 4, 4
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Flatten()
            )
        self.fc1 = nn.Linear(64*4*4, self.latent_dim)
        self.fc2 = nn.Linear(64*4*4, self.latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64*4*4),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, 3, stride=3, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh())

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        x.permute(0, 3, 1, 2)
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return [self.decoder(z), mu, logvar]
    
    def loss(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy((recon_x+1)/2, (x+1)/2, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
        # self.loss_dict['reconstruction_loss'].append(BCE.item())
        # self.loss_dict['kl_divergence'].append(KLD.item())
        return BCE + KLD

    def train_VAE(self, trainloader, optimizer, epochs=10):
        loss_history = []
        # self.train()
        for epoch in range(epochs):
            for batch_idx, (data, _) in enumerate(trainloader):
                data = data.to(device)
                
                optimizer.zero_grad()
                recon_batch, mu, logvar = self(data)
                loss = self.loss(recon_batch, data, mu, logvar)
                loss.backward()
                loss_history.append(loss.item())
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    clear_output(wait=True)
                    plt.figure(figsize=(10, 10))
                    plt.plot(loss_history, label='loss')
                    plt.legend()
                    plt.xlabel('batch')
                    plt.ylabel('loss')
                    plt.title(f'Epoch: {epoch}, step: {batch_idx}/{len(trainloader)}\nloss: {loss.item():.3f}')
                    plt.show()

    def encode(self, img):
        self.eval()
        with torch.no_grad():
            lat = self.encoder(img.to(device))
            z, *_ = self.bottleneck(lat)
        return z

    def decode(self, lat):
        self.eval()
        with torch.no_grad():
            rec = self.decoder(lat)
        return rec
