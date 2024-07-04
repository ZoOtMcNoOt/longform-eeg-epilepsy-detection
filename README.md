# Longform EEG Epilepsy Detection Project

## Overview

This project analyzes electroencephalography (EEG) data to identify key features such as mutual information, recurrence rate, and laminarity. The main scripts included in this project are designed to preprocess the data, calculate the necessary metrics, and save the results for further analysis. The primary goal is to use Recurrence Quantification Analysis (RQA) to identify potential markers for epilepsy.

## Project Structure

1. **main.py**: This is the main script that orchestrates the entire process. It reads the ABF file, processes each channel, downsamples the data, and performs the RQA.

2. **data_processing.py**: Contains functions for reading the ABF file, applying a low-pass filter, and downsampling the data.

3. **segment_processing.py**: Handles the segmentation of the data and calls the relevant functions for calculating time delay, recurrence rates, and laminarity.

4. **knn_mutual_information.py**: Implements the k-nearest neighbors (k-NN) algorithm to calculate mutual information and determine the optimal time delay.

5. **rqa_analysis.py**: Contains functions for performing RQA, finding the appropriate radii for the desired recurrence rates, and calculating laminarity.

6. **save_results.py**: A utility script for saving the results to a CSV file.

## Key Algorithms and Concepts

### k-NN Mutual Information
The k-NN algorithm is used to estimate mutual information, which helps determine the optimal time delay. This is crucial for reconstructing the phase space of the EEG data.

### Recurrence Quantification Analysis (RQA)
RQA is a method used to quantify the number and duration of recurrences of a dynamical system. It involves:
- **Recurrence Rate (REC)**: Measures the density of recurrence points.
- **Laminarity (LAM)**: Indicates the presence of laminar states where the system's state changes slowly.

### Binary Search for Radius
A binary search algorithm is employed to find the radii corresponding to 1% and 5% recurrence rates efficiently. This avoids the need to test every possible radius, saving computational time.

### Normalization
Before performing RQA, the time series data is normalized to ensure that the recurrence matrix is scaled appropriately. This helps in accurately finding the radii and calculating the recurrence rates.

## How to Use

1. **Install Dependencies**: Make sure you have the required Python packages installed. The primary dependencies are `pyabf`, `numpy`, `scipy`, `torch`, and `pyrqa`.

2. **Run the Main Script**: Execute the `main.py` script to start the analysis. You can specify the ABF file path and other parameters directly in the script.

3. **Check Results**: The results will be saved to a CSV file named `rqa_results.csv`. This file will contain details of the time delay, recurrence rates, radii, and laminarity for each epoch and embedding dimension.

## Example

Here's a step-by-step example of how the scripts work together:

1. **main.py**: Orchestrates the entire process.
2. **data_processing.py**: Reads the ABF file and downsamples the data.
3. **segment_processing.py**: Segments the data into epochs and processes each segment.
4. **knn_mutual_information.py**: Calculates the mutual information to find the optimal time delay.
5. **rqa_analysis.py**: Finds the appropriate radii, normalizes the time series, and calculates laminarity.
6. **save_results.py**: Saves the processed results to a CSV file.

## Detailed Print Statements

To better understand the intermediate steps, print statements have been added to the scripts. These will output the progress and results of the binary search for radii, the time delay calculation, and the normalization process.

## Versions

- **Python**: 3.10.14
- **CUDA**: 11.8
- **PyTorch**: Compatible with CUDA 11.8
- **pyabf**, **numpy**, **scipy**, **pyrqa**: Latest versions as of 2024.
