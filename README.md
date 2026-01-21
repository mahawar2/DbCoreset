# Desity-based Coreset Algorithm (DbCoreset)

Official PyTorch implementation of the paper:
**DBCoreset: A Density-based Coreset Algorithm for Computationally Efficient Chest X-ray Diagnosis**

---

## Pipeline

1. Train a vanilla convolutional autoencoder on chest X-ray images
2. Extract latent features using the trained encoder
3. Generate a coreset using the proposed LCG algorithm
4. Create a coreset dataset
5. Train EfficientNet-B0 classifier on the coreset
6. Evaluate on the full test set

---

## Folder Structure

```
degsity_based_coreset/
│
├── data/
│   ├── csv_dataset.py
│   └── coreset_dataset.py
│
├── autoencoder/
│   ├── model.py
│   ├── train.py
│   └── validate.py
│
├── dbcoreset/
│   └── dbcoreset.py
│
├── classifier/
│   ├── model.py
│   ├── train.py
│   └── validate.py
│
├── utils/
│   └── metrics.py
│
├── train_autoencoder.py
├── generate_dbcoreset.py  
├── train_classifier.py
├── test_classifier.py
│
├── README.md
├── requirements.txt
└── LICENSE

```

---

## CSV Format

Your CSV file must look like:

```
image_path,label_0,label_1,label_2,...,label_C
```

Each label must be 0/1 (multi-label supported).

---

## Installation

```
pip install -r requirements.txt
```

---

---

## Full Training & Evaluation Pipeline

### 1. Train Autoencoder
```bash
python train_autoencoder.py --csv train.csv --val_csv val.csv
```
This trains the vanilla convolutional autoencoder and saves:
```
autoencoder_best.pth
```

---

### 2. Generate Coreset using DbCoreset
```bash
python generate_dbcoreset.py \
  --csv train.csv \
  --ae_ckpt autoencoder_best.pth \
  --budget 5000 \
  --eps 0.005 \
  --minpts 10 \
  --lambda1 0.7 \
  --lambda2 0.3

```
This will generate:
```
dbcoreset_indices.npy
```

---

### 3. Train Classifier on Coreset
```bash
python train_classifier.py --train_csv train.csv --val_csv val.csv --indices dbcoreset_indices.npy
```
This trains EfficientNet-B0 on the coreset and saves:
```
classifier_best.pth
```

---

### 4. Test on Full Test Set
```bash
python test_classifier.py --test_csv test.csv --ckpt classifier_best.pth
```

---
