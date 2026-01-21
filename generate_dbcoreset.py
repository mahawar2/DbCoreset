import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader


from data.csv_dataset import ChestXrayCSVDataset
from autoencoder.model import ConvAutoencoder
from lcg.feature_extractor import extract_features
from dbcoreset.dbcoreset import DbCoreset






def main(args):
device = 'cuda' if torch.cuda.is_available() else 'cpu'


dataset = ChestXrayCSVDataset(args.csv)
loader = DataLoader(dataset, batch_size=32, shuffle=False)


model = ConvAutoencoder(latent_dim=32)
model.load_state_dict(torch.load(args.ae_ckpt, map_location=device))
model = model.to(device)


print("Extracting features...")
X, _ = extract_features(model, loader, device)


print("Running DbCoreset...")
dbc = DbCoreset(
budget=args.budget,
eps=args.eps,
minpts=args.minpts,
lambda1=args.lambda1,
lambda2=args.lambda2
)


indices = dbc.run(X)


print(f"DbCoreset size: {len(indices)}")


np.save(args.out_indices, indices)




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, required=True)
parser.add_argument('--ae_ckpt', type=str, required=True)
parser.add_argument('--budget', type=int, required=True)
parser.add_argument('--eps', type=float, default=0.005)
parser.add_argument('--minpts', type=int, default=10)
parser.add_argument('--lambda1', type=float, default=0.7)
parser.add_argument('--lambda2', type=float, default=0.3)
parser.add_argument('--out_indices', type=str, default='dbcoreset_indices.npy')


args = parser.parse_args()
main(args)
