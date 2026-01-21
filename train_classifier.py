import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn

from data.coreset_dataset import CoresetDataset
from classifier.model import EfficientNetClassifier
from classifier.train import train_classifier
from classifier.validate import validate_classifier


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    indices = np.load(args.indices)

    train_dataset = CoresetDataset(args.train_csv, indices)
    val_dataset = CoresetDataset(args.val_csv, indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    num_classes = train_dataset.labels.shape[1]

    model = EfficientNetClassifier(num_classes=num_classes).to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0

    for epoch in range(50):
        train_loss = train_classifier(model, train_loader, optimizer, criterion, device)
        val_loss, metrics = validate_classifier(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, AUC={metrics['mean_auc']:.4f}")

        if metrics['mean_auc'] > best_auc:
            best_auc = metrics['mean_auc']
            torch.save(model.state_dict(), 'classifier_best.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--indices', type=str, required=True)
    args = parser.parse_args()

    main(args)
