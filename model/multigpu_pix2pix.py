import os
import torch
from generator import Generator
from discriminator import Discriminator
from utils import VecDataset
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
from torch import nn
import cv2
import random

# Hyperparameters
batch_size = 16
epochs = 200
lr = 0.001

# Load and split dataset
print('Loading dataset...')
dataset = VecDataset('./datasets/vector')
length = len(dataset)
trainset, testset = random_split(dataset, [int(0.9 * length), length - int(0.9 * length)])

# Create data loaders
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=batch_size // 2, shuffle=True, num_workers=4)

# Initialize models and move to GPU
generator = Generator(in_channels=2, out_channels=1).cuda()
discriminator = Discriminator(in_channels=2 + 1).cuda()

# Optimizers
optimizer_G = AdamW(generator.parameters(), lr=lr)
optimizer_D = AdamW(discriminator.parameters(), lr=lr)

# Loss functions
bce_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()


def lossfuc(out, label, alpha=100):
    """Custom loss function that emphasizes errors in high-value regions"""
    return torch.mean((out - label) ** 2 * torch.log(2 + alpha * label))


def flip_and_rot(x, y):
    """Apply random flips and rotations for data augmentation"""
    if random.random() > 0.5:
        x = torch.flip(x, dims=[3])
        y = torch.flip(y, dims=[3])
    if random.random() > 0.5:
        x = torch.flip(x, dims=[2])
        y = torch.flip(y, dims=[2])

    # Random rotation
    rotation_angle = random.choice([0, 1, 2, 3])
    x = torch.rot90(x, k=rotation_angle, dims=[2, 3])
    y = torch.rot90(y, k=rotation_angle, dims=[2, 3])

    return x, y


# Training loop
print('Begin training...')
for e in range(epochs):
    # Training phase
    generator.train()
    discriminator.train()
    with tqdm(enumerate(trainloader), total=len(trainloader)) as pbar:
        for i, (x, y) in pbar:
            x, y = x.cuda(), y.cuda()
            x, y = flip_and_rot(x, y)

            # Train discriminator
            optimizer_D.zero_grad()
            y_fake = generator(x)
            D_real = discriminator(x, y)
            D_fake = discriminator(x, y_fake.detach())
            D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
            D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            optimizer_D.step()

            # Train generator
            optimizer_G.zero_grad()
            D_fake = discriminator(x, y_fake)
            if e < 100 or random.random() > 0.5:
                G_loss = lossfuc(y_fake, y)  # Focus on reconstruction first
            else:
                G_loss = lossfuc(y_fake, y) + 0.0001 * bce_loss(D_fake, torch.ones_like(D_fake))

            G_loss.backward()
            optimizer_G.step()

            pbar.set_description(f"Epoch {e} Loss G: {G_loss.item():.4f} Loss D: {D_loss.item():.4f}")

    torch.cuda.empty_cache()

    # Evaluation phase
    generator.eval()
    discriminator.eval()
    counter = 0
    with torch.no_grad(), tqdm(enumerate(testloader), total=len(testloader)) as pbar:
        for i, (x, y) in pbar:
            x, y = x.cuda(), y.cuda()
            x, y = flip_and_rot(x, y)
            y_fake = generator(x)
            D_fake = discriminator(x, y_fake)
            G_loss = lossfuc(y_fake, y) + 0.001 * bce_loss(D_fake, torch.ones_like(D_fake))
            pbar.set_description(f"Test Loss: {G_loss.item():.4f}")

            # Save test cases for visualization
            y = y.detach().cpu().numpy()
            y_fake = y_fake.detach().cpu().numpy()
            x = x.detach().cpu().numpy()
            for i in range(y.shape[0]):
                case_folder = f'./test_case_pix2pix/case_{counter + i}'
                os.makedirs(case_folder, exist_ok=True)
                cv2.imwrite(f'{case_folder}/data1.png', x[i][0] * 255)
                cv2.imwrite(f'{case_folder}/data2.png', x[i][1] * 255)
                cv2.imwrite(f'{case_folder}/label.png', y[i, 0] * 255)
                cv2.imwrite(f'{case_folder}/output.png', y_fake[i, 0] * 255)
            counter += y.shape[0]

        print(f'Saving checkpoint at epoch {e}, testing loss is {G_loss.item():.4f}')

    # Save model checkpoint
    torch.save(generator, f'./checkpoints/saved_pix2pix.pt')
    torch.cuda.empty_cache()
