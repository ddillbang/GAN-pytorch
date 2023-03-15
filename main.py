import torch
import torch.nn as nn
from src.DataLoader import dataLoader
from model import GAN
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
import os

EPOCHS = 200
BATCH_SIZE = 128
LR = 0.0002
DATASET = 'MNIST'
# DATASET = 'cifar10'

noise_size = 128

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using : {device}")

train_dataloader, test_dataloader = dataLoader(DATASET, BATCH_SIZE)

generator = GAN.Generator(noise_size, 28 if DATASET == 'MNIST' else 32).to(device)
discriminator = GAN.Discriminator(28 if DATASET == 'MNIST' else 32).to(device)

g_optim = torch.optim.Adam(generator.parameters(), LR)
d_optim = torch.optim.Adam(discriminator.parameters(), LR)
criterion = nn.BCELoss()

#train
def train():
    for epoch in range(EPOCHS):
        for i, (img, label) in enumerate(train_dataloader):
            n_bt_size = len(img)
            real_label = torch.full((n_bt_size, 1), 1, dtype=torch.float32).to(device)
            fake_label = torch.full((n_bt_size, 1), 0, dtype=torch.float32).to(device)

            real_img = img.reshape(len(img), -1).to(device)

            g_optim.zero_grad()

            z = torch.randn(n_bt_size, noise_size).to(device)
            fake_img = generator(z)

            g_loss = criterion(discriminator(fake_img), real_label)

            g_loss.backward()
            g_optim.step()

            d_optim.zero_grad()

            z = torch.randn(n_bt_size, noise_size).to(device)
            fake_img = generator(z)
            fake_loss = criterion(discriminator(fake_img), fake_label)
            real_loss = criterion(discriminator(real_img), real_label)
            d_loss = (fake_loss + real_loss) / 2

            d_loss.backward()
            d_optim.step()
            
            d_perform = discriminator(real_img).mean()
            g_perform = discriminator(fake_img).mean()

            if (i + 1) % 150 == 0:
                print(f"Epoch [ {epoch}/{EPOCHS} ],  Step [ {i+1}/{len(train_dataloader)} ],  d_loss : {d_loss.item()},  g_loss : {g_loss.item()}")

        print(f"Epoch {epoch} discriminator performance : {d_perform}, generator perform: {g_perform}")
        
        #print imgs
        z = torch.randn(25, noise_size).to(device)
        img = generator(z)
        img = 0.5 * img + 5
        img = img.reshape(25, 28, 28).data.cpu().numpy()

        fig, axs = plt.subplots(5, 5)
        count = 0
        for j in range(5):
            for k in range(5):
                axs[j, k].imshow(img[count])
                axs[j, k].axis('off')
                count += 1
        
        plt.show()
        print(i)

if __name__ == '__main__':
    train()