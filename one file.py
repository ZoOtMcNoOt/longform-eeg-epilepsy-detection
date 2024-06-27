import torch
import cupy as cp
import pandas as pd
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
import timeit
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
    """Find the first local minimum in the mutual information values."""
    local_minima_indices = [i for i in range(1, len(mi_values) - 1) if mi_values[i] < mi_values[i - 1] and mi_values[i] < mi_values[i + 1]]
    return local_minima_indices[0] if local_minima_indices else 0

def select_time_delay_optimized(data, max_tau, n_jobs=-1):
    """Select the appropriate time delay based on mutual information."""
    def compute_mi(tau):
        return knn_mutual_information(data[:-tau], data[tau:])
    
    mi_values = Parallel(n_jobs=n_jobs)(delayed(compute_mi)(tau) for tau in range(1, max_tau + 1))
    return find_first_local_minimum(mi_values) + 1

def calculate_percent_rec(dist_matrix, rad):
    """Calculate percentage of recurrence points in the recurrence matrix."""
    recurrence_matrix = dist_matrix <= rad
    percent_rec = cp.sum(recurrence_matrix) / dist_matrix.size * 100
    return percent_rec

def select_rad(dist_matrix, rad_values):
    """Select the appropriate RAD based on 1.0% and 5.0% REC values."""
    percent_recs = cp.array([calculate_percent_rec(dist_matrix, rad) for rad in rad_values])
    rad_for_1_percent = rad_values[cp.searchsorted(percent_recs, cp.array([1.0]), side='left')]
    rad_for_5_percent = rad_values[cp.searchsorted(percent_recs, cp.array([5.0]), side='left')]
    selected_rad = (rad_for_1_percent + rad_for_5_percent) / 2
    return selected_rad

def run_rqa(data, embedding_dim, time_delay, rad):
    """Run Recurrence Quantification Analysis (RQA) using pyRQA."""
    time_series = TimeSeries(cp.asnumpy(data), embedding_dimension=embedding_dim, time_delay=time_delay)
    settings = Settings(time_series, neighbourhood=FixedRadius(rad), similarity_measure=EuclideanMetric, theiler_corrector=1)
    rqa = RQAComputation.create(settings)
    results = rqa.run()
    results.min_diagonal_line_length = 5
    return results.laminarity

# Generate test sample data with 4800 data points
epoch_data = cp.sin(cp.linspace(0, 20 * cp.pi, 4800))  # Example sine wave data

# Define parameters
embedding_dimensions = [5, 9, 13, 17, 21, 25]
max_tau = 50  # Adjust based on your data
rad_values = cp.linspace(0.01, 1.0, 100)  # Adjust range as needed

# Calculate laminarity for the test sample
epoch_length = 2400  # For example, if each epoch is 2400 data points
laminarity_results = []

time_delay = select_time_delay_optimized(cp.asnumpy(epoch_data), max_tau)

# Distance matrix calculation
dist_matrix = squareform(pdist(cp.asnumpy(epoch_data.reshape(-1, 1)), 'euclidean'))
dist_matrix = cp.asarray(dist_matrix)
selected_rad = select_rad(dist_matrix, rad_values)

# Convert rad_values and selected_rad to NumPy arrays for pyRQA
rad_values = rad_values.get()
selected_rad = selected_rad.get()

# Calculate laminarity for each embedding dimension
for embedding_dim in embedding_dimensions:
    laminarity = run_rqa(epoch_data.get(), embedding_dim, time_delay, selected_rad)
    laminarity_results.append({
        "Epoch": 1,
        "Embedding Dimension": embedding_dim,
        "Laminarity": laminarity
    })

# Convert results to a DataFrame and save to CSV
df = pd.DataFrame(laminarity_results)
# Define a proper path to save the CSV file
output_path = os.path.join(os.path.expanduser('~'), 'laminarity_results.csv')
df.to_csv(output_path, index=False)
print(f"Laminarity results saved to {output_path}")

# Benchmarking
def benchmark_mutual_information():
    return knn_mutual_information(cp.asnumpy(epoch_data[:-1]), cp.asnumpy(epoch_data[1:]))

def benchmark_find_first_local_minimum():
    mi_values = [knn_mutual_information(cp.asnumpy(epoch_data[:-tau]), cp.asnumpy(epoch_data[tau:])) for tau in range(1, max_tau + 1)]
    return find_first_local_minimum(mi_values)

def benchmark_select_time_delay():
    return select_time_delay_optimized(cp.asnumpy(epoch_data), max_tau)

def benchmark_calculate_percent_rec():
    dist_matrix = squareform(pdist(cp.asnumpy(epoch_data.reshape(-1, 1)), 'euclidean'))
    dist_matrix = cp.asarray(dist_matrix)
    return calculate_percent_rec(dist_matrix, 0.5)

def benchmark_select_rad():
    dist_matrix = squareform(pdist(cp.asnumpy(epoch_data.reshape(-1, 1)), 'euclidean'))
    dist_matrix = cp.asarray(dist_matrix)
    rad_values_cp = cp.asarray(rad_values)  # Ensure rad_values is in the correct format
    rad_values_np = rad_values_cp.get()
    return select_rad(dist_matrix, cp.asarray(rad_values_np))

def benchmark_run_rqa():
    dist_matrix = squareform(pdist(cp.asnumpy(epoch_data.reshape(-1, 1)), 'euclidean'))
    dist_matrix = cp.asarray(dist_matrix)
    selected_rad = select_rad(dist_matrix, cp.asarray(rad_values))
    time_delay = select_time_delay_optimized(cp.asnumpy(epoch_data), max_tau)
    return run_rqa(cp.asnumpy(epoch_data), embedding_dimensions[0], time_delay, selected_rad.get())

# Run the benchmarks
time_mutual_information = timeit.timeit(benchmark_mutual_information, number=10)
time_find_first_local_minimum = timeit.timeit(benchmark_find_first_local_minimum, number=10)
time_select_time_delay = timeit.timeit(benchmark_select_time_delay, number=10)
time_calculate_percent_rec = timeit.timeit(benchmark_calculate_percent_rec, number=10)
time_select_rad = timeit.timeit(benchmark_select_rad, number=10)
time_run_rqa = timeit.timeit(benchmark_run_rqa, number=10)

print(f"Mutual Information time: {time_mutual_information:.6f} seconds")
print(f"Find First Local Minimum time: {time_find_first_local_minimum:.6f} seconds")
print(f"Select Time Delay time: {time_select_time_delay:.6f} seconds")
print(f"Calculate Percent Recurrence time: {time_calculate_percent_rec:.6f} seconds")
print(f"Select RAD time: {time_select_rad:.6f} seconds")
print(f"Run RQA time: {time_run_rqa:.6f} seconds")
