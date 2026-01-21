import torch
from tqdm import tqdm

def validate_autoencoder(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            recon, _ = model(images)
            loss = criterion(recon, images)
            total_loss += loss.item()

    return total_loss / len(dataloader)
