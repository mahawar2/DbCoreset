import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

class ChestXrayCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.image_paths = self.df.iloc[:, 0].values
        self.labels = self.df.iloc[:, 1:].values.astype('float32')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0
        image = image[None, :, :]  # (1, H, W)

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image), torch.tensor(label)
