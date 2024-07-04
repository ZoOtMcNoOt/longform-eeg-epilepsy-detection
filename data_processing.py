import pyabf
import numpy as np
from scipy.signal import butter, filtfilt

def read_abf_data(file_path):
    abf = pyabf.ABF(file_path)
    data = abf.data[0]  # Assuming we are working with the first channel
    sampling_rate = abf.dataRate
    return data, sampling_rate

def low_pass_filter(data, cutoff_freq, sampling_rate):
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def downsample_data(data, original_rate, target_rate):
    downsample_factor = int(original_rate / target_rate)
    downsampled_data = data[::downsample_factor]
    return downsampled_data, target_rate
