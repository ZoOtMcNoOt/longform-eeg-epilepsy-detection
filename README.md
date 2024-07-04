# Longform EEG Epilepsy Detection

This project aims to process and analyze longform EEG data to detect epilepsy-related features using Recurrence Quantification Analysis (RQA). The analysis involves downsampling the EEG data, calculating mutual information to determine the time delay, and performing RQA to extract laminarity metrics.

## Project Structure

- **main.py**: The main script that orchestrates the entire process from loading the data to saving the results.
- **segment_processing.py**: Contains functions for processing each segment of the data and downsampling.
- **knn_mutual_information.py**: Contains functions related to calculating mutual information and selecting the optimal time delay.
- **rqa_analysis.py**: Contains functions for performing RQA, finding radii, normalizing the time series, and calculating laminarity.
- **save_results.py**: Contains the function to save results to a CSV file.

## Requirements

- Python 3.10.14
- CUDA 11.8
- PyTorch 1.12.1+cu116
- PyABF (latest version)
- NumPy 1.22.4
- SciPy 1.8.1
- PyRQA (latest version)
- CSV (built-in Python library)
