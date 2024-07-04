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

def select_time_delay(data, max_tau, k=3):
    """
    Select the optimal time delay for phase space reconstruction using mutual information.

    Parameters:
    data (array-like): Input data series.
    max_tau (int): Maximum time delay to consider.
    k (int): Number of nearest neighbors for mutual information calculation.

    Returns:
    int: Optimal time delay.
    """
    data = np.array(data)
    previous_mi = knn_mutual_information(data, np.roll(data, 1), k)
    print("\nCalculating mutual information with adaptive step size:")
    for tau in range(2, max_tau + 1):
        mi = knn_mutual_information(data, np.roll(data, tau), k)
        print(f"Time delay {tau}: MI = {mi:.6f}")
        if mi > previous_mi:
            print(f"First local minimum found at time delay: {tau - 1}\n")
            return tau - 1
        previous_mi = mi
    print(f"Max time delay reached: {max_tau}\n")
    return max_tau
