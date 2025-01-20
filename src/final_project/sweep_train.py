from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import yaml
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data import load_data, accuracy
from MLOPS_GROUP_33.models.models import GAT, SpGAT


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

current_dir = os.path.dirname(__file__)  # train.py 的路径
project_root = os.path.abspath(os.path.join(current_dir, '..'))
processed_data_dir = os.path.join(project_root, 'config', 'sweep_config.yaml')
with open(processed_data_dir, "r") as file:
    sweep_config = yaml.safe_load(file)

sweep_id = wandb.sweep(sweep_config, project="gat_sweep_project")
def train_sweep(config=None):
    with wandb.init(config=config):

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        current_dir = os.path.dirname(__file__)  # train.py 的路径
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        processed_data_dir = os.path.join(project_root, 'MLOPS_GROUP_33', 'data', 'processed\\')

        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = load_data(path=processed_data_dir)

        # Model and optimizer
        if args.sparse:
            model = SpGAT(nfeat=features.shape[1], 
                        nhid=wandb.config.hidden, 
                        nclass=int(labels.max()) + 1, 
                        dropout=args.dropout, 
                        nheads=args.nb_heads, 
                        alpha=args.alpha)
        else:
            model = GAT(nfeat=features.shape[1], 
                        nhid=wandb.config.hidden, 
                        nclass=int(labels.max()) + 1, 
                        dropout=wandb.config.dropout, 
                        nheads=args.nb_heads, 
                        alpha=args.alpha)
        optimizer = optim.Adam(model.parameters(), 
                            lr=wandb.config.lr, 
                            weight_decay=args.weight_decay)

        if args.cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

        features, adj, labels = Variable(features), Variable(adj), Variable(labels)


        # Train model
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = float('inf')  
        best_epoch = 0

        config = wandb.config

        best_val_acc = 0
        for epoch in range(args.epochs):

            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if not args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output = model(features, adj)

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'acc_train: {:.4f}'.format(acc_train.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val.data.item()),
                  'time: {:.4f}s'.format(time.time() - t))
            
            loss_values.append(loss_val.data.item())

            torch.save(model.state_dict(), '{}.pkl'.format(epoch))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                break

            if acc_val.item() > best_val_acc:
                best_val_acc = acc_val.item()

            files = glob.glob('*.pkl')
            for file in files:
                epoch_nb = int(file.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)
        wandb.run.summary["val_accuracy"] = best_val_acc
        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

wandb.agent(sweep_id, train_sweep, count=10)

