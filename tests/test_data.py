import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from tests import _PATH_DATA
from final_project.data import load_data, accuracy
import torch 

#Test the data loading
def test_data_loading():
    """Test the data loading and its properties."""
    dataset_name = "cora"  # Dataset name to test
    data_path = os.path.join(_PATH_DATA, "processed")  # Keep 'processed' in the path

    # Ensure the processed data path exists
    assert os.path.exists(data_path), f"Data path {data_path} does not exist!"

    # Ensure required files exist
    content_file = os.path.join(data_path, f"{dataset_name}.content")
    cites_file = os.path.join(data_path, f"{dataset_name}.cites")
    assert os.path.exists(content_file), f"Content file {content_file} not found!"
    assert os.path.exists(cites_file), f"Cites file {cites_file} not found!"

    # Load the data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path=f"{data_path}/", dataset=dataset_name)

    # Assert basic properties of the loaded data
    assert adj.shape[0] == adj.shape[1], "Adjacency matrix should be square!"
    assert features.shape[0] == labels.shape[0], "Number of features and labels must match!"
    assert len(idx_train) > 0, "Training indices should not be empty!"
    assert len(idx_val) > 0, "Validation indices should not be empty!"
    assert len(idx_test) > 0, "Test indices should not be empty!"

#Test the data accuracy
def test_accuracy():
    # Test case 1: Perfect accuracy
    output = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.6, 0.2, 0.2]])
    labels = torch.tensor([2, 1, 0])

    # Expected output: 1.0 (100% accuracy)
    acc = accuracy(output, labels).float()
    assert torch.isclose(acc, torch.tensor(1.0).float()), f"Expected accuracy 1.0, got {acc}"

    # Test case 2: Some incorrect predictions
    output = torch.tensor([[0.9, 0.1, 0.0], [0.3, 0.4, 0.3], [0.6, 0.2, 0.2]])
    labels = torch.tensor([0, 1, 0])

    # Expected output: 1.0 (since all predictions are correct)
    acc = accuracy(output, labels).float()
    assert torch.isclose(acc, torch.tensor(1.0).float()), f"Expected accuracy 1.0, got {acc}"

    # Test case 3: All incorrect predictions (adjusted for floating-point errors)
    output = torch.tensor([[0.1, 0.1, 0.8], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    labels = torch.tensor([0, 0, 1])

    # Expected output: 0.0 (no correct predictions)
    acc = accuracy(output, labels).float()
    assert torch.isclose(acc, torch.tensor(0.0).float(), atol=1e-4), f"Expected accuracy 0.0, got {acc}"

