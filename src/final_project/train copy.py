from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from data import load_data, accuracy
from lighting_models import GATLightningModule
import pytorch_lightning as pl
parser = argparse.ArgumentParser()
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--max_epochs', type=int, default=10000, help='Maximum number of epochs.')
args = parser.parse_args()

# Load data
current_dir = os.path.dirname(__file__)  
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
processed_data_dir = os.path.join(project_root, 'MLOPS_GROUP_33', 'data', 'processed\\')
adj, features, labels, idx_train, idx_val, idx_test = load_data(path=processed_data_dir)

# Move data to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
features, adj, labels = features.to(device), adj.to(device), labels.to(device)
idx_train, idx_val, idx_test = idx_train.to(device), idx_val.to(device), idx_test.to(device)
print(features,adj)
# Prepare dataset
dataset = TensorDataset(features, labels)
val_dataset = TensorDataset(features, labels)
test_dataset = TensorDataset(features, labels)

train_loader = DataLoader(dataset, batch_size=1)
val_loader = DataLoader(val_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)

# Initialize model
model = GATLightningModule(
    nfeat=features.shape[1],
    nhid=args.hidden,
    nclass=int(labels.max()) + 1,
    dropout=args.dropout,
    alpha=args.alpha,
    nheads=args.nb_heads,
    adj=adj,
    idx_train=idx_test,
    lr=args.lr,
    weight_decay=args.weight_decay,
    sparse=args.sparse

)



# Trainer
trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    log_every_n_steps=1,
    accelerator="auto",
    devices="auto",
)

# Train and test
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(model, test_dataloaders=test_loader)