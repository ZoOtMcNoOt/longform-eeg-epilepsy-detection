import torch
import cupy as cp
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
import timeit

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
    # Compute the differences
    diffs = cp.diff(mi_values)
    # Find where the slope changes from negative to positive
    local_minima = cp.where((diffs[:-1] < 0) & (diffs[1:] > 0))[0] + 1
    return int(local_minima[0].get()) if len(local_minima) > 0 else 0

def select_time_delay_optimized(data, max_tau):
    """Select the appropriate time delay based on mutual information using CuPy and PyTorch."""
    data_cp = cp.asarray(data)
    taus = cp.arange(1, max_tau + 1)

    # Prepare the shifted arrays using a CuPy kernel to avoid implicit conversions
    def prepare_shifted_arrays(tau):
        return data_cp[:-tau], data_cp[tau:]

    shifted_arrays = [prepare_shifted_arrays(tau) for tau in taus]
    x_taus, y_taus = zip(*shifted_arrays)

    # Calculate mutual information for each tau in parallel
    def compute_mi(x, y):
        return knn_mutual_information(cp.asnumpy(x), cp.asnumpy(y))
    
    mi_values = cp.array(Parallel(n_jobs=-1)(delayed(compute_mi)(x, y) for x, y in zip(x_taus, y_taus)))
    
    # Adding 1 to the result to get the correct time delay
    return find_first_local_minimum(mi_values) + 1

def calculate_percent_rec(dist_matrix, rad):
    """Calculate percentage of recurrence points in the recurrence matrix."""
    recurrence_matrix = cp.less_equal(dist_matrix, rad)
    percent_rec = cp.sum(recurrence_matrix) / dist_matrix.size * 100
    return percent_rec

def select_rad(dist_matrix, rad_values):
    """Select the appropriate RAD based on 1.0% and 5.0% REC values using CuPy."""
    # Ensure dist_matrix and rad_values are CuPy arrays
    dist_matrix = cp.asarray(dist_matrix)
    rad_values = cp.asarray(rad_values)

    # Calculate percent recurrence for each rad value in parallel
    percent_recs = cp.array([calculate_percent_rec(dist_matrix, rad) for rad in rad_values])
    
    # Use CuPy's searchsorted for fast searching
    rad_for_1_percent = rad_values[cp.searchsorted(percent_recs, cp.array([1.0]))]
    rad_for_5_percent = rad_values[cp.searchsorted(percent_recs, cp.array([5.0]))]
    
    # Calculate the average of the two rad values
    selected_rad = (rad_for_1_percent + rad_for_5_percent) / 2
    
    return selected_rad

def compute_laminarity(data, embedding_dim, time_delay, rad):
    """Compute laminarity for given data using custom RQA implementation."""
    # Embed the time series
    print(embedding_dim, time_delay, rad)
    N = len(data) - (embedding_dim - 1) * time_delay
    embedded_data = cp.zeros((N, embedding_dim))
    
    for i in range(embedding_dim):
        embedded_data[:, i] = data[i * time_delay : i * time_delay + N]

    # Compute the distance matrix directly in CuPy
    dist_matrix = cp.asarray(squareform(pdist(cp.asnumpy(embedded_data), metric='euclidean')))

    # Create the recurrence matrix
    rad_cp = cp.asarray(rad)
    recurrence_matrix = cp.less_equal(dist_matrix, rad_cp)

    # Compute vertical lines in the recurrence matrix
    vertical_lines = cp.sum(recurrence_matrix, axis=1)
    number_of_vertical_lines_points = cp.sum(vertical_lines[vertical_lines >= 2])
    number_of_recurrence_points = cp.sum(recurrence_matrix)

    # Calculate laminarity
    laminarity = number_of_vertical_lines_points / number_of_recurrence_points if number_of_recurrence_points > 0 else 0

    return laminarity

# Generate test sample data with 4800 data points
epoch_data = cp.sin(cp.linspace(0, 20 * cp.pi, 2400))  # Example sine wave data

# Define parameters
embedding_dimensions = [5, 9, 13, 17, 21, 25]
max_tau = 50  # Adjust based on your data
rad_values = cp.linspace(0.001, 1.0, 1000)  # Adjust range as needed


# Calculate laminarity for the test sample
epoch_length = 2400  # For example, if each epoch is 2400 data points
laminarity_results = []

time_delay = select_time_delay_optimized(cp.asnumpy(epoch_data), max_tau)

# Distance matrix calculation
epoch_data_reshaped = epoch_data.reshape(-1, 1)
dist_matrix = cp.asarray(squareform(pdist(cp.asnumpy(epoch_data_reshaped), 'euclidean')))
selected_rad = select_rad(dist_matrix, rad_values)

# Convert rad_values and selected_rad to NumPy arrays for pyRQA
rad_values = rad_values.get()
selected_rad = selected_rad.get()

# Calculate laminarity for each embedding dimension
for embedding_dim in embedding_dimensions:
    laminarity = compute_laminarity(epoch_data, embedding_dim, time_delay, selected_rad)
    laminarity_results.append({
        "Epoch": 1,
        "Embedding Dimension": embedding_dim,
        "Laminarity": laminarity
    })

# Convert results to a DataFrame and save to CSV
df = pd.DataFrame(laminarity_results)
# Save the DataFrame to a CSV file in the current working directory
df.to_csv('laminarity_results-3.csv', index=False)
print("Laminarity results saved to laminarity_results.csv")

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

def benchmark_compute_laminarity():
    dist_matrix = squareform(pdist(cp.asnumpy(epoch_data.reshape(-1, 1)), 'euclidean'))
    dist_matrix = cp.asarray(dist_matrix)
    selected_rad = select_rad(dist_matrix, cp.asarray(rad_values))
    time_delay = select_time_delay_optimized(cp.asnumpy(epoch_data), max_tau)
    return compute_laminarity(epoch_data, embedding_dimensions[0], time_delay, selected_rad.get())

# Run the benchmarks
time_mutual_information = timeit.timeit(benchmark_mutual_information, number=120)
time_find_first_local_minimum = timeit.timeit(benchmark_find_first_local_minimum, number=120)
time_select_time_delay = timeit.timeit(benchmark_select_time_delay, number=120)
time_calculate_percent_rec = timeit.timeit(benchmark_calculate_percent_rec, number=120)
time_select_rad = timeit.timeit(benchmark_select_rad, number=120)
time_compute_laminarity = timeit.timeit(benchmark_compute_laminarity, number=120)

print(f"Mutual Information time: {time_mutual_information:.6f} seconds")
print(f"Find First Local Minimum time: {time_find_first_local_minimum:.6f} seconds")
print(f"Select Time Delay time: {time_select_time_delay:.6f} seconds")
print(f"Calculate Percent Recurrence time: {time_calculate_percent_rec:.6f} seconds")
print(f"Select RAD time: {time_select_rad:.6f} seconds")
print(f"Compute Laminarity time: {time_compute_laminarity:.6f} seconds")
