# mnist.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Models
# -----------------------

class MLPGenerator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class DCGANGenerator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 128, 7, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))


class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# -----------------------
# Training
# -----------------------

def train(model_type="gan", epochs=50, z_dim=100, save_path="models", batch_size=128, lr=0.0002):
    os.makedirs(save_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    if model_type == "gan":
        G = MLPGenerator(z_dim).to(device)
        D = MLPDiscriminator().to(device)
    else:
        G = DCGANGenerator(z_dim).to(device)
        D = DCGANDiscriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, z_dim, device=device)

    for epoch in range(epochs):
        for real, _ in dataloader:
            real = real.to(device)
            b_size = real.size(0)
            noise = torch.randn(b_size, z_dim, device=device)

            real_labels = torch.ones(b_size, 1, device=device)
            fake_labels = torch.zeros(b_size, 1, device=device)

            # Train Discriminator
            fake = G(noise)
            d_real = D(real)
            d_fake = D(fake.detach())
            d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # Train Generator
            fake = G(noise)
            g_loss = criterion(D(fake), real_labels)
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        print(f"[{model_type.upper()}] Epoch [{epoch+1}/{epochs}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                fakes = G(fixed_noise).cpu()
            save_image(fakes, f"{save_path}/{model_type}_epoch{epoch+1}.png", normalize=True, nrow=8)

    torch.save(G.state_dict(), f"{save_path}/{model_type}_G.pth")
    torch.save(D.state_dict(), f"{save_path}/{model_type}_D.pth")
    print("✅ Models saved!")


# -----------------------
# Generate Images
# -----------------------

def generate(model_type="gan", z_dim=100, save_path="models"):
    if model_type == "gan":
        G = MLPGenerator(z_dim).to(device)
    else:
        G = DCGANGenerator(z_dim).to(device)

    G.load_state_dict(torch.load(f"{save_path}/{model_type}_G.pth", map_location=device))
    G.eval()

    noise = torch.randn(64, z_dim, device=device)
    with torch.no_grad():
        fake = G(noise).cpu()
    grid = make_grid(fake, nrow=8, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"Generated {model_type.upper()} Images")
    plt.axis("off")
    plt.show()


# -----------------------
# Main Entry
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gan", "dcgan"], required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    if args.train:
        train(model_type=args.model)
    elif args.load:
        generate(model_type=args.model)
