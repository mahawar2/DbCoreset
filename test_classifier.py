import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn

from data.csv_dataset import ChestXrayCSVDataset
from classifier.model import EfficientNetClassifier
from classifier.validate import validate_classifier


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_dataset = ChestXrayCSVDataset(args.test_csv)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    num_classes = test_dataset.labels.shape[1]

    model = EfficientNetClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    criterion = nn.BCEWithLogitsLoss()

    loss, metrics = validate_classifier(model, test_loader, criterion, device)

    print("Test Results:")
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()

    main(args)
