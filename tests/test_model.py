import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
import pytest
import torch.nn.functional as F
from final_project.models import GAT  # Adjust if necessary

def test_forward_pass():
    # Test parameters
    nfeat = 10    # Number of input features
    nhid = 20     # Number of hidden units
    nclass = 3    # Number of output classes
    dropout = 0.5
    alpha = 0.2   # LeakyReLU negative slope
    nheads = 4    # Number of attention heads

    # Initialize model
    model = GAT(nfeat, nhid, nclass, dropout, alpha, nheads)

    # Create a dummy input tensor (batch size of 5, feature size of 10)
    x = torch.randn(5, nfeat)

    # Create a dummy adjacency matrix (5 nodes, 5 nodes)
    adj = torch.randn(5, 5)  # In practice, the adjacency matrix would be sparse or a binary matrix

    # Forward pass
    model.train()  # Set model to training mode
    output = model(x, adj)  # Get the output

    # Check if the output has the correct shape
    assert output.shape == (5, nclass), f"Expected output shape (5, {nclass}), got {output.shape}"

    # Convert log probabilities to probabilities using softmax
    probs = F.softmax(output, dim=1)

    # Ensure that the sum of probabilities is approximately 1 for each instance in the batch
    assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-6), \
        "The sum of probabilities for each instance should be approximately 1."

    # Ensure that the model is properly applying dropout (output should not be exactly the same as input)
    x_no_dropout = model.forward(x, adj)
    assert not torch.equal(output, x_no_dropout), "Dropout was not applied correctly."
