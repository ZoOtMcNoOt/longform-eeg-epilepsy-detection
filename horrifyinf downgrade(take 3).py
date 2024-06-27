import torch
import cupy as cp
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
import pandas as pd
import os

def knn_mutual_information(x, y):
    """Calculate mutual information for continuous variables using PyTorch."""
    x = torch.tensor(x, dtype=torch.float32).cuda()
    y = torch.tensor(y, dtype=torch.float32).cuda()
    xy = torch.stack([x, y], dim=1)
    
    n_neighbors = 3
    knn = torch.cdist(xy, xy, p=2.0)
    knn, _ = torch.topk(knn, n_neighbors + 1, largest=False)
    knn = knn[:, -1]
    
    hx = torch.mean(torch.log(knn)) + torch.log(torch.tensor(len(x), dtype=torch.float32).cuda()) - torch.digamma(torch.tensor(n_neighbors, dtype=torch.float32).cuda())
    hy = torch.mean(torch.log(knn)) + torch.log(torch.tensor(len(y), dtype=torch.float32).cuda()) - torch.digamma(torch.tensor(n_neighbors, dtype=torch.float32).cuda())
    
    hxy = torch.mean(torch.log(knn)) + torch.log(torch.tensor(len(x), dtype=torch.float32).cuda()) - torch.digamma(torch.tensor(n_neighbors, dtype=torch.float32).cuda())
    
    mi = hx + hy - hxy
    return mi.item()

def find_first_local_minimum(mi_values):
    """Find the first local minimum in the mutual information values using vectorized operations with CuPy."""
    mi_values = cp.asarray(mi_values)
    diffs = cp.diff(mi_values)
    local_minima = cp.where((diffs[:-1] < 0) & (diffs[1:] > 0))[0] + 1
    return int(local_minima[0].get()) if len(local_minima) > 0 else 0

def select_time_delay_optimized(data, max_tau):
    """Select the appropriate time delay based on mutual information using CuPy and PyTorch."""
    data_cp = cp.asarray(data)
    taus = cp.arange(1, max_tau + 1)
    
    x_taus = [data_cp[:-tau].get() for tau in taus]
    y_taus = [data_cp[tau:].get() for tau in taus]
    
    def compute_mi(x, y):
        return knn_mutual_information(x, y)
    
    mi_values = cp.array(Parallel(n_jobs=-1)(delayed(compute_mi)(x, y) for x, y in zip(x_taus, y_taus)))
    return find_first_local_minimum(mi_values) + 1

def calculate_percent_rec(dist_matrix, rad):
    """Calculate percentage of recurrence points in the recurrence matrix."""
    recurrence_matrix = cp.less_equal(dist_matrix, rad)
    percent_rec = cp.sum(recurrence_matrix) / dist_matrix.size * 100
    return percent_rec

def select_rad(dist_matrix, rad_values):
    """Select the appropriate RAD based on 1.0% and 5.0% REC values using CuPy."""
    dist_matrix = cp.asarray(dist_matrix)
    rad_values = cp.asarray(rad_values)

    percent_recs = cp.array([calculate_percent_rec(dist_matrix, rad) for rad in rad_values])
    rad_for_1_percent = rad_values[cp.searchsorted(percent_recs, cp.asarray(1.0))]
    rad_for_5_percent = rad_values[cp.searchsorted(percent_recs, cp.asarray(5.0))]
    selected_rad = (rad_for_1_percent + rad_for_5_percent) / 2
    
    return selected_rad

def embed_time_series(data, embedding_dim, time_delay):
    """Embed the time series data in higher-dimensional space using time delays."""
    N = len(data) - (embedding_dim - 1) * time_delay
    embedded_data = cp.zeros((N, embedding_dim), dtype=cp.float32)
    for i in range(embedding_dim):
        embedded_data[:, i] = data[i * time_delay : i * time_delay + N]
    return embedded_data

def compute_distance_matrix(embedded_data):
    """Compute the pairwise distance matrix and normalize by the maximum distance."""
    dist_matrix = cp.asarray(squareform(pdist(cp.asnumpy(embedded_data), metric='euclidean')))
    max_distance = cp.max(dist_matrix)
    normalized_dist_matrix = dist_matrix / max_distance
    return normalized_dist_matrix, max_distance

def construct_recurrence_matrix(dist_matrix, rad_percentage, theiler_window):
    """Construct the binary recurrence matrix."""
    threshold = rad_percentage / 100.0  # Convert percentage to a fraction
    recurrence_matrix = cp.less_equal(dist_matrix, threshold)
    
    # Apply the Theiler window by setting the diagonal elements within the window to False
    for i in range(-theiler_window, theiler_window + 1):
        if i != 0:  # Avoid setting the main diagonal to False
            diag_indices = cp.arange(max(0, i), min(dist_matrix.shape[0], dist_matrix.shape[0] + i))
            recurrence_matrix[diag_indices, diag_indices - i] = False
    
    return recurrence_matrix

def compute_laminarity(recurrence_matrix):
    """Compute the laminarity from the recurrence matrix."""
    num_recurrence_points = cp.sum(recurrence_matrix).astype(cp.float32)
    print("Number of recurrence points:", num_recurrence_points)  # Debugging

    vertical_lines = cp.zeros_like(recurrence_matrix, dtype=cp.float32)

    # Count vertical lines by checking columns of the recurrence matrix
    for i in range(recurrence_matrix.shape[1]):
        column = recurrence_matrix[:, i]
        count = 0
        for j in range(len(column)):
            if column[j]:
                count += 1
            else:
                if count >= 2:
                    vertical_lines[j - count:j, i] = 1  # Mark vertical line positions
                count = 0
        if count >= 2:
            vertical_lines[-count:, i] = 1

    print("Vertical lines matrix:", vertical_lines)  # Debugging

    num_vertical_lines_points = cp.sum(vertical_lines * recurrence_matrix).astype(cp.float32)
    print("Number of vertical lines points:", num_vertical_lines_points)  # Debugging

    laminarity = num_vertical_lines_points / num_recurrence_points if num_recurrence_points > 0 else 0
    return laminarity

def run_rqa(data, embedding_dim, time_delay, rad_percentage, theiler_window):
    """Run Recurrence Quantification Analysis (RQA) to compute laminarity."""
    embedded_data = embed_time_series(data, embedding_dim, time_delay)
    dist_matrix, max_distance = compute_distance_matrix(embedded_data)
    recurrence_matrix = construct_recurrence_matrix(dist_matrix, rad_percentage, theiler_window)
    laminarity = compute_laminarity(recurrence_matrix)
    return laminarity

# Example usage
epoch_data = cp.sin(cp.linspace(0, 20 * cp.pi, 1200))  # Example sine wave data
embedding_dim = 5
time_delay = 1
max_tau = 50  # Adjust based on your data
rad_values = cp.linspace(0.01, 1.0, 100)  # Adjust range as needed

# Calculate the time delay
time_delay = select_time_delay_optimized(cp.asnumpy(epoch_data), max_tau)
print(f"Selected Time Delay: {time_delay}")

# Embed the time series and compute the distance matrix
embedded_data = embed_time_series(epoch_data, embedding_dim, time_delay)
dist_matrix, max_distance = compute_distance_matrix(embedded_data)

# Select the appropriate RAD
selected_rad = select_rad(dist_matrix, rad_values)
print(f"Selected RAD: {selected_rad}")

# Calculate the Theiler window
theiler_window = (embedding_dim - 1) * time_delay
print(f"Theiler Window: {theiler_window}")

# Convert selected_rad to percentage of max_distance
rad_percentage = (selected_rad.get() / max_distance.get()) * 100
print(f"RAD Percentage: {rad_percentage}")

# Calculate laminarity for each embedding dimension
laminarity_results = []
for embedding_dim in [5, 9, 13, 17, 21, 25]:
    laminarity = run_rqa(epoch_data, embedding_dim, time_delay, rad_percentage, theiler_window)
    laminarity_results.append({
        "Epoch": 1,
        "Embedding Dimension": embedding_dim,
        "Laminarity": laminarity
    })

# Convert results to a DataFrame and save to CSV
df = pd.DataFrame(laminarity_results)
df.to_csv('laminarity_results.csv', index=False)
print("Laminarity results saved to laminarity_results.csv")