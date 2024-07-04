import numpy as np
import torch
from scipy.special import digamma

def to_tensor(data, use_cuda=True):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    return torch.tensor(data, dtype=torch.float32, device=device)

def knn_mutual_information(x, y, k=3):
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
    data = np.array(data)
    previous_mi = knn_mutual_information(data, np.roll(data, 1), k)
    for tau in range(2, max_tau + 1):
        mi = knn_mutual_information(data, np.roll(data, tau), k)
        if mi > previous_mi:
            return tau - 1
        previous_mi = mi
    return max_tau
