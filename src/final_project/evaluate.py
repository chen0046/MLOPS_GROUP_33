from contextlib import asynccontextmanager
import torch
import io
import scipy.sparse as sp
import numpy as np
from models import GAT
from fastapi import FastAPI, File, UploadFile
from PIL import Image



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global  state_dict,device
    # Training settings
    print("Loading model")
    device = torch.device("cpu")
    state_dict = torch.load("/gcs/mlops-33-busket/processed/", map_location=torch.device(device))
    yield
    print("Cleaning up")
    del state_dict,device


app = FastAPI(lifespan=lifespan)


@app.post("/test/")
async def caption(data1: UploadFile = File(..., media_type="content"),
                  data2: UploadFile = File(..., media_type="cities")):

    content1 = await data1.read()  
    content2 = await data2.read()  
    content1_io = io.BytesIO(content1)
    content2_io = io.BytesIO(content2)
    idx_features_labels = np.genfromtxt(content1_io, dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(content2_io, dtype=np.int32,delimiter="\t")

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])


    """Generate a caption for an image."""
    model = GAT(nfeat=features.shape[1], 
                    nhid=8, 
                    nclass=int(labels.max()) + 1, 
                    dropout=0.6, 
                    nheads=8, 
                    alpha=0.2)
    model.eval() 
    model.load_state_dict(state_dict)
    with torch.no_grad():
        output = model(features, adj)
    _, predicted_class = torch.max(output, dim=1)
    return predicted_class.tolist()


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot