import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from models import SpGAT, GAT
from data import load_data, accuracy


class GATLightningModule(pl.LightningModule):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj, idx_train, lr, weight_decay, sparse=False):
        super().__init__()
        self.save_hyperparameters()
        self.adj=adj
        self.idx_train=idx_train
        if sparse:
            self.model = SpGAT(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=dropout, nheads=nheads, alpha=alpha)
        else:
            self.model = GAT(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=dropout, nheads=nheads, alpha=alpha)

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x,adj):
        return self.model(x, adj)

    def training_step(self, batch, batch_idx):

        x, labels= batch
        output = self(x, self.adj)
        loss = F.nll_loss(output[self.idx_train], labels[self.idx_train])
        acc = accuracy(output[self.idx_train], labels[self.idx_train])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        x, labels = batch
        output = self(x, self.adj)
        loss = F.nll_loss(output[self.idx_val], labels[self.idx_val])
        acc = accuracy(output[self.idx_val], labels[self.idx_val])
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):

        x, labels = batch
        output = self(x, self.adj)
        loss = F.nll_loss(output[self.idx_test], labels[self.idx_test])
        acc = accuracy(output[self.idx_test], labels[self.idx_test])
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_accuracy', acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
