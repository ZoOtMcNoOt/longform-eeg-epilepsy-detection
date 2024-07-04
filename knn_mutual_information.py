import numpy as np
import torch
from scipy.special import digamma

def to_tensor(data, use_cuda=True):
    """
    Convert data to a PyTorch tensor.

    Parameters:
    data (array-like): Input data to be converted.
    use_cuda (bool): Flag to use CUDA (GPU) if available.

    Returns:
    torch.Tensor: Converted tensor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    return torch.tensor(data, dtype=torch.float32, device=device)

def knn_mutual_information(x, y, k=3):
    """
    Calculate the k-nearest neighbors mutual information between two variables.

    Parameters:
    x (array-like): First variable.
    y (array-like): Second variable.
    k (int): Number of nearest neighbors.

    Returns:
    float: Mutual information value.
    """
    x, y = to_tensor(x), to_tensor(y)
    xy = torch.stack([x, y], dim=1)
    knn = torch.cdist(xy, xy, p=2.0)
    knn, _ = torch.topk(knn, k + 1, largest=False)
    knn = knn[:, -1]
    knn = torch.clamp(knn, min=1e-10)
    hx = torch.mean(torch.log(knn)) + torch.log(torch.tensor(len(x), dtype=torch.float32)) - digamma(k)
    hy = torch.mean(torch.log(knn)) + torch.log(torch.tensor(len(y), dtype=torch.float32)) - digamma(k)
    hxy = torch.mean(torch.log(knn)) + torch.log(torch.tensor(len(x), dtype=torch.float32)) - digamma(k)
    mi = hx + hy - hxy
    return mi

def select_time_delay(data, max_tau):
    """
    Select the time delay using the first local minimum of mutual information.
    """
    data = to_tensor(data)
    mi_values = []
    
    for tau in range(1, max_tau + 1):
        mi = knn_mutual_information(data[:-tau], data[tau:])
        mi_values.append((tau, mi.item()))
        print(f"Time delay {tau}: MI = {mi.item()}")
    
    mi_values = np.array(mi_values)
    
    # Finding the first local minimum
    for i in range(1, len(mi_values) - 1):
        if mi_values[i - 1, 1] > mi_values[i, 1] < mi_values[i + 1, 1]:
            min_idx = i
            print(f"First local minimum found at time delay: {mi_values[min_idx, 0]}")
            return int(mi_values[min_idx, 0])
    
    # If no local minimum is found, return the time delay with the overall minimum MI
    min_idx = np.argmin(mi_values[:, 1])
    print(f"No local minimum found, using global minimum at time delay: {mi_values[min_idx, 0]}")
    return int(mi_values[min_idx, 0])
