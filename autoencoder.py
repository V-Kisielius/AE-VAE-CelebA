import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output

from config import device

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), # batch_size, 16, 64, 64
            nn.MaxPool2d(2, stride=2), # batch_size, 16, 32, 32
            nn.ELU(), 
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # batch_size, 32, 16, 16
            nn.MaxPool2d(2, stride=2), # batch_size, 8, 8, 8
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # batch_size, 64, 4, 4
            nn.Flatten(),
            nn.ELU(),
            nn.Linear(64*4*4, latent_dim),
            nn.Tanh()
            )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64*4*4),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, 3, stride=3, padding=2),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh())

    def forward(self, x):
        x.permute(0, 3, 1, 2)
        lat = self.encoder(x)
        rec = self.decoder(lat)
        return rec

    def train_autoencoder(self, trainloader, num_epochs=10, learning_rate=1e-3):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        loss_history = []
        for epoch in range(num_epochs):
            for i, batch in enumerate(trainloader):
                img, _ = batch
                clear_output(wait=True)

                img = img.to(device)
                output = self(img)
                loss = criterion(output, img)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())

                if (i + 1) % 100 == 0:
                    plt.figure(figsize=(15, 10))
                    plt.plot(loss_history)
                    plt.xlabel('Step')
                    plt.ylabel('Loss')
                    plt.title(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}')
                    plt.show()
        return self

    def encode(self, img):
        self.eval()
        with torch.no_grad():
            lat = self.encoder(img.to(device))
        return lat

    def decode(self, lat):
        self.eval()
        with torch.no_grad():
            rec = self.decoder(lat)
        return rec