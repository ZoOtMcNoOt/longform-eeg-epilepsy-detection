import pyabf
import numpy as np
from scipy.signal import butter, filtfilt
from pyunicorn.timeseries.recurrence_plot import RecurrencePlot
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool

# Define a bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Recurrence Quantification Analysis (RQA)
def compute_rqa(epoch, embedding_dim, time_delay):
    rp = RecurrencePlot(epoch, embedding_dim, time_delay)
    rec_rate = rp.recurrence_rate()
    laminarity = rp.laminarity()
    return rec_rate, laminarity

# Determine optimal embedding dimension
def determine_optimal_embedding_dimension(epochs):
    embedding_dims = range(5, 26, 4)
    rqa_results = {dim: [] for dim in embedding_dims}

    for epoch in epochs:
        for dim in embedding_dims:
            rec_rate, laminarity = compute_rqa(epoch, dim, time_delay)
            rqa_results[dim].append((rec_rate, laminarity))
    
    # Analyze results to determine optimal embedding dimension
    # Placeholder: assume a fixed optimal embedding dimension
    optimal_embedding_dim = 10
    return optimal_embedding_dim

# Process a single file
def process_file(file_path):
    print(f"Processing file: {file_path}")
    
    # Load ABF file
    abf = pyabf.ABF(file_path)
    data = abf.data
    sampling_rate = abf.dataRate

    # Apply bandpass filter
    filtered_data = bandpass_filter(data, lowcut=0.5, highcut=40, fs=sampling_rate)

    # Segment data into epochs
    epoch_length = 12 * sampling_rate
    epochs = [filtered_data[i:i+epoch_length] for i in range(0, len(filtered_data), epoch_length)]

    # Determine optimal embedding dimension
    optimal_embedding_dim = determine_optimal_embedding_dimension(epochs)

    # Compute RQA for each epoch using the optimal embedding dimension
    rqa_results = [compute_rqa(epoch, optimal_embedding_dim, time_delay=1) for epoch in epochs]

    # Plot results
    rec_rates, laminarities = zip(*rqa_results)
    plt.figure()
    plt.plot(rec_rates, label='Recurrence Rate')
    plt.plot(laminarities, label='Laminarity')
    plt.xlabel('Epoch')
    plt.ylabel('RQA Measures')
    plt.legend()
    plt.title(f"RQA Measures for {os.path.basename(file_path)}")
    plt.show()

    return rqa_results

# Main function to process multiple files
def main(folder_path):
    # Gather all ABF files in the folder
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.abf')]
    
    # Process files concurrently
    with Pool(processes=len(file_paths)) as pool:
        results = pool.map(process_file, file_paths)

if __name__ == "__main__":
    # Folder path containing the ABF files
    folder_path = "D:\\DKO-2023-24\\D2 1mm TBI control\\abf-files"
    main(folder_path)
