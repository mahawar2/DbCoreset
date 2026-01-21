import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import compute_metrics


def validate_classifier(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = torch.sigmoid(model(images))
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    metrics = compute_metrics(all_labels, all_preds)

    return total_loss / len(dataloader), metrics
