import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

from data.csv_dataset import ChestXrayCSVDataset
from autoencoder.model import ConvAutoencoder
from autoencoder.train import train_autoencoder
from autoencoder.validate import validate_autoencoder


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = ChestXrayCSVDataset(args.csv)
    val_dataset = ChestXrayCSVDataset(args.val_csv)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ConvAutoencoder(latent_dim=32).to(device)

    optimizer = Adam(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.25)
    criterion = nn.MSELoss()

    best_val = float('inf')

    for epoch in range(50):
        train_loss = train_autoencoder(model, train_loader, optimizer, criterion, device)
        val_loss = validate_autoencoder(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'autoencoder_best.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    args = parser.parse_args()

    main(args)
