import pyabf
from segment_processing import process_segment
from save_results import save_results_to_csv
import os

def read_abf_file(file_path):
    """
    Read an ABF file and return the data and the original sampling rate.

    Parameters:
    file_path (str): Path to the ABF file.

    Returns:
    np.ndarray: Data from the ABF file.
    int: Original sampling rate of the data.
    """
    abf = pyabf.ABF(file_path)
    return abf, abf.dataRate

def downsample_data(data, original_rate, target_rate):
    """
    Downsample the data to a target rate.

    Parameters:
    data (np.ndarray): Input data to downsample.
    original_rate (int): Original sampling rate of the data.
    target_rate (int): Target downsampling rate.

    Returns:
    np.ndarray: Downsampled data.
    """
    factor = original_rate // target_rate
    return data[::factor]

def process_file(file_path, output_folder, target_sampling_rate, epoch_duration_seconds, embedding_dims):
    # Read the ABF file
    abf, original_sampling_rate = read_abf_file(file_path)
    
    for channel in range(abf.channelCount):
        abf.setSweep(0, channel=channel)
        channel_data = abf.sweepY
        
        # Downsample the data
        downsampled_data = downsample_data(channel_data, original_sampling_rate, target_sampling_rate)
        
        epoch_samples = int(epoch_duration_seconds * target_sampling_rate)
        total_epochs = len(downsampled_data) // epoch_samples
        
        results = []
        for epoch_index in range(total_epochs):
            segment_start = epoch_index * epoch_samples
            segment_end = segment_start + epoch_samples
            
            if segment_end > len(downsampled_data):
                break
            
            print(f"\nProcessing File: {file_path}, Channel: {channel + 1}, Epoch: {epoch_index + 1}")
            epoch_results = process_segment(downsampled_data, segment_start, segment_end, target_sampling_rate, embedding_dims, epoch_index, os.path.basename(file_path))
            results.extend(epoch_results)
        
        # Save results for this channel
        output_file_path = os.path.join(output_folder, f"channel_{channel}_results.csv")
        save_results_to_csv(results, output_file_path)
