import pyabf
import numpy as np
from scipy.signal import butter, filtfilt

def read_abf_file(file_path):
    """
    Read an ABF file and return the data.

    Parameters:
    file_path (str): Path to the ABF file.

    Returns:
    np.ndarray: Data from the ABF file.
    """
    abf = pyabf.ABF(file_path)
    return abf.data

def low_pass_filter(data, sampling_rate, cutoff=40):
    """
    Apply a low-pass filter to the data.

    Parameters:
    data (np.ndarray): Input data to filter.
    sampling_rate (int): Sampling rate of the data.
    cutoff (int): Cutoff frequency for the low-pass filter.

    Returns:
    np.ndarray: Filtered data.
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

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
