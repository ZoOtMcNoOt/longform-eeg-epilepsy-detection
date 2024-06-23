### pip install neo numba cupy tqdm pandas matplotlib

import neo
import numpy as np
import cupy as cp  # Use cupy for GPU acceleration
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from scipy.signal import butter, filtfilt
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Use cupy for GPU acceleration
xp = cp

# Define preprocessing and analysis functions
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

@jit(nopython=True)
def compute_rqa_metrics_vectorized(epoch, embedding_dim, time_delay):
    n = len(epoch)
    embedded_data = np.array([epoch[i:i + embedding_dim * time_delay:time_delay] for i in range(n - embedding_dim * time_delay + 1)])
    distance_matrix = np.linalg.norm(embedded_data[:, None, :] - embedded_data[None, :, :], axis=2)
    
    recurrence_matrix = (distance_matrix < embedding_dim).astype(np.int32)
    recurrence_rate = np.sum(recurrence_matrix) / recurrence_matrix.size
    
    # Calculate laminarity
    vertical_lines = np.diff(recurrence_matrix, axis=0) == 0
    laminarity = np.sum(vertical_lines) / np.sum(recurrence_matrix)
    
    return recurrence_rate, laminarity

def load_and_preprocess_abf_file(file_path, cutoff, fs, selected_channels):
    reader = neo.io.AxonIO(filename=file_path)
    block = reader.read_block(lazy=False)
    segment = block.segments[0]
    data = []
    for channel in selected_channels:
        analogsignal = segment.analogsignals[channel]
        data.append(analogsignal.as_array().flatten())
    data = np.array(data).T
    filtered_data = butter_lowpass_filter(data, cutoff, fs)
    return filtered_data

# Parameters for preprocessing
cutoff = 40.0  # Hz
fs = 400.0  # Sampling frequency

# Define a dictionary to map animal IDs to their respective channels and date changes
animal_channel_map = {
    'Fi-9-Epi': ([0, 1], None),
    'Fi-13': ([2], None),
    'Fi-4': ([3], None),
    'Fi-2-Epi': ([4, 5, 6], None),
    'Fi-3-Epi': ([7], datetime(2023, 10, 10)),
    'Fi-3-Epi (C/HC)': ([8, 9, 10], None),  # Assuming this for C/HC
    'Fi-5': ([11, 12], None),
    'Fi-12': ([13], datetime(2023, 10, 24)),
    'Fi-6': ([14], datetime(2023, 10, 11)),
    'Fi-7': ([15], None)
}

# Helper function to determine the correct animal ID based on date
def get_animal_id(animal_id, date):
    if animal_id == 'Fi-3-Epi' and date > datetime(2023, 10, 10):
        return 'Fi-11'
    if animal_id == 'Fi-6' and date > datetime(2023, 10, 11):
        return 'Fi-8'
    return animal_id

def process_file(file_path, cutoff, fs):
    # Extract the date from the filename or another source
    # Here assuming the filename contains the date, e.g., '2023_09_06_0007.abf'
    filename = os.path.basename(file_path)
    date_str = filename.split('_')[0:3]
    date = datetime.strptime('_'.join(date_str), '%Y_%m_%d')
    
    processed_data = []
    
    for animal_id, (channels, change_date) in animal_channel_map.items():
        current_animal_id = get_animal_id(animal_id, date)
        
        # Load and preprocess the file
        data = load_and_preprocess_abf_file(file_path, cutoff, fs, channels)
        processed_data.append(data)
    
    return xp.concatenate(processed_data, axis=0)

# Directory containing the .abf files
abf_files_directory = 'path_to_your_abf_files_directory'

# Get a list of all .abf files in the directory
abf_files = [os.path.join(abf_files_directory, f) for f in os.listdir(abf_files_directory) if f.endswith('.abf')]

# Concatenate data from all files in parallel
concatenated_data = []

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_file, file_path, cutoff, fs) for file_path in abf_files]
    
    for future in tqdm(as_completed(futures), total=len(futures)):
        concatenated_data.append(future.result())

concatenated_data = xp.concatenate(concatenated_data, axis=0)

# Segment data into 24-hour epochs with 1-hour overlap
epoch_length = 24 * 3600 * fs  # 24 hours in samples
overlap = 1 * 3600 * fs  # 1 hour in samples
epochs = [concatenated_data[i:i+epoch_length] for i in range(0, len(concatenated_data), epoch_length - overlap)]

# Compute RQA metrics for each epoch
embedding_dims = [5, 9, 13, 17, 21, 25]
time_delay = 1  # Needs to be determined
rqa_metrics = []

for epoch in epochs:
    epoch_metrics = []
    for dim in embedding_dims:
        rec, lam = compute_rqa_metrics_vectorized(epoch.get(), dim, time_delay)
        epoch_metrics.append((rec, lam))
    rqa_metrics.append(epoch_metrics)

def estimate_embedding_dimension(rqa_metrics, embedding_dims):
    est_eds = []
    for metrics in rqa_metrics:
        variations = [lam for _, lam in metrics]
        diffs = np.diff(variations)
        est_dim = embedding_dims[np.argmax(diffs) + 1]  # +1 because np.diff reduces the array size by 1
        est_eds.append(est_dim)
    return est_eds

est_eds = estimate_embedding_dimension(rqa_metrics, embedding_dims)

# Plotting estimated embedding dimensions over time
plt.figure(figsize=(12, 6))
plt.plot(est_eds, marker='o', linestyle='-')
plt.title('Estimated Embedding Dimension Over Time')
plt.xlabel('Epoch')
plt.ylabel('Estimated Embedding Dimension')
plt.grid(True)
plt.show()

# Save the concatenated data to a CSV file (optional)
data_df = pd.DataFrame(xp.asnumpy(concatenated_data), columns=[f'Channel_{ch}' for ch in range(concatenated_data.shape[1])])
data_df.to_csv('concatenated_ecog_data.csv', index=False)

# Save the estimated embedding dimensions to a CSV file (optional)
est_eds_df = pd.DataFrame(est_eds, columns=['Estimated Embedding Dimension'])
est_eds_df.to_csv('estimated_embedding_dimensions.csv', index=False)

# Display the DataFrame to the user
print(est_eds_df.head())
