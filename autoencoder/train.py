import torch
from tqdm import tqdm

def train_autoencoder(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, _ in tqdm(dataloader):
        images = images.to(device)

        optimizer.zero_grad()
        recon, _ = model(images)
        loss = criterion(recon, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
