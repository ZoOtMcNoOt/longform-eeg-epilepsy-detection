import numpy as np
import pyabf

def read_abf_data(file_path):
    abf = pyabf.ABF(file_path)
    sampling_rate = abf.dataRate
    data = abf.sweepY
    return data, sampling_rate

def low_pass_filter(data, cutoff_freq, sampling_rate):
    from scipy.signal import butter, filtfilt
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data
