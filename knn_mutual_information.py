import torch
from scipy.special import digamma
import numpy as np

def to_tensor(data, use_cuda=True):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    return torch.tensor(data.copy(), dtype=torch.float32, device=device)

def knn_mutual_information(x, y, k=3):
    x, y = to_tensor(x), to_tensor(y)
    xy = torch.stack([x, y], dim=1)
    knn = torch.cdist(xy, xy, p=2.0)  # Euclidean distance
    knn, _ = torch.topk(knn, k + 1, largest=False)
    knn = knn[:, -1]

    knn = torch.clamp(knn, min=1e-10)

    hx = torch.mean(torch.log(knn)) + torch.log(torch.tensor(len(x), dtype=torch.float32, device=knn.device)) - digamma(k)
    hy = torch.mean(torch.log(knn)) + torch.log(torch.tensor(len(y), dtype=torch.float32, device=knn.device)) - digamma(k)
    hxy = torch.mean(torch.log(knn)) + torch.log(torch.tensor(len(x), dtype=torch.float32, device=knn.device)) - digamma(k)

    mi = hx + hy - hxy
    return mi.item()

def mutual_information(data, tau, k=3):
    return knn_mutual_information(data[:-tau], data[tau:], k)

def select_time_delay(data, max_tau=100, k=3, initial_step=5, min_step=1, max_step=20, rate_threshold=0.003):
    mi_values = []
    step_size = initial_step

    print("Calculating mutual information with adaptive step size:")
    tau = 1
    previous_mi = mutual_information(data, tau, k)
    mi_values.append(previous_mi)
    print(f"Time delay {tau}: MI = {previous_mi}")

    while tau + step_size <= max_tau:
        tau += step_size
        current_mi = mutual_information(data, tau, k)
        mi_values.append(current_mi)
        print(f"Time delay {tau}: MI = {current_mi}")

        if len(mi_values) > 1:
            delta_mi = abs(current_mi - mi_values[-2])
            if delta_mi < rate_threshold:
                step_size = min(max_step, step_size * 2)
            else:
                step_size = max(min_step, step_size // 2)
        
        if len(mi_values) > 2 and mi_values[-3] > mi_values[-2] < mi_values[-1]:
            print(f"First local minimum found at time delay: {tau - step_size - 1}")
            return tau - step_size - 1

        previous_mi = current_mi

    print("No local minimum found within the given max_tau.")
    return tau
