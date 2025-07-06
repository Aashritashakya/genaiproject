import os
import zipfile
import requests
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import streamlit as st
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Parameters
z_dim = 100
num_classes = 2
img_size = 128
channels = 3
batch_size = 32
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset download and preparation
def download_and_prepare_dataset():
    url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    zip_path = "cats_and_dogs_filtered.zip"
    extract_dir = "cats_and_dogs_filtered"
    dataset_dir = "CatDog_Dataset"

    if not os.path.exists(zip_path):
        st.write("Downloading dataset...")
        r = requests.get(url, stream=True)
        with open(zip_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        st.write("Download complete.")

    if not os.path.exists(extract_dir):
        st.write("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        st.write("Extraction complete.")

    os.makedirs(dataset_dir, exist_ok=True)
    cat_src = os.path.join(extract_dir, "cats_and_dogs_filtered", "train", "cats")
    dog_src = os.path.join(extract_dir, "cats_and_dogs_filtered", "train", "dogs")
    cat_dst = os.path.join(dataset_dir, "0")
    dog_dst = os.path.join(dataset_dir, "1")
    os.makedirs(cat_dst, exist_ok=True)
    os.makedirs(dog_dst, exist_ok=True)

    def copy_images(src, dst, max_images=1000):
        count = 0
        for fname in os.listdir(src):
            if fname.endswith(".jpg"):
                shutil.copy(os.path.join(src, fname), os.path.join(dst, fname))
                count += 1
                if count >= max_images:
                    break
        return count

    if len(os.listdir(cat_dst)) == 0:
        copied_cats = copy_images(cat_src, cat_dst)
        st.write(f"Copied {copied_cats} cat images.")

    if len(os.listdir(dog_dst)) == 0:
        copied_dogs = copy_images(dog_src, dog_dst)
        st.write(f"Copied {copied_dogs} dog images.")

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, z_dim)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim * 2, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, channels, 4, 4, 0),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([noise, label_embedding], dim=1)
        x = x.view(-1, z_dim * 2, 1, 1)
        return self.model(x)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        self.conv = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (img_size // 8) * (img_size // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_map = self.label_emb(labels).view(-1, 1, img_size, img_size)
        if img.size(2) != img_size:
            img = nn.functional.interpolate(img, size=(img_size, img_size))
        x = torch.cat([img, label_map], dim=1)
        x = self.conv(x)
        return self.fc(x)

# Training function
def train():
    download_and_prepare_dataset()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = datasets.ImageFolder("CatDog_Dataset", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    st.write("Starting training...")
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size_i = imgs.size(0)

            valid = torch.ones(batch_size_i, 1).to(device)
            fake = torch.zeros(batch_size_i, 1).to(device)

            optimizer_G.zero_grad()
            z = torch.randn(batch_size_i, z_dim).to(device)
            gen_labels = torch.randint(0, num_classes, (batch_size_i,), device=device)
            gen_imgs = generator(z, gen_labels)
            g_loss = criterion(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(imgs, labels), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        st.write(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    torch.save(generator.state_dict(), "generator_catdog.pth")
    st.success("Training complete! Model saved as 'generator_catdog.pth'.")

@st.cache_resource(show_spinner=False)
def load_generator():
    model = Generator().to(device)
    try:
        model.load_state_dict(torch.load("generator_catdog.pth", map_location=device))
        st.success("Loaded pretrained generator weights.")
    except FileNotFoundError:
        st.error("No pretrained generator weights found! Please train the model first.")
    model.eval()
    return model

def main():
    st.title("Conditional GAN: Cat vs Dog")

    mode = st.radio("Choose mode:", ["Train", "Generate"])

    if mode == "Train":
        if st.button("Start Training"):
            train()
    else:
        generator = load_generator()
        label_map = {0: "Cat", 1: "Dog"}
        choice = st.selectbox("Generate image of:", list(label_map.values()))
        label_idx = 0 if choice == "Cat" else 1
        num_samples = st.slider("Number of images", 1, 16, 4)

        if st.button("Generate"):
            with torch.no_grad():
                noise = torch.randn(num_samples, z_dim).to(device)
                labels = torch.full((num_samples,), label_idx, dtype=torch.long).to(device)
                gen_imgs = generator(noise, labels).cpu()
                gen_imgs = (gen_imgs + 1) / 2

                grid_img = make_grid(gen_imgs, nrow=4)
                np_img = grid_img.permute(1, 2, 0).numpy()

                fig, ax = plt.subplots(figsize=(6,6))
                ax.axis("off")
                ax.imshow(np_img)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
