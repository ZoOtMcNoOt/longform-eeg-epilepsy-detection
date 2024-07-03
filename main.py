from data_processing import read_abf_data, low_pass_filter
from segment_processing import process_segment
from save_results import save_results_to_csv

if __name__ == '__main__':
    # Path to the ABF file
    abf_file_path = "C:\\Users\\grant\\OneDrive - Texas A&M University\\D2 1mm TBI control\\abf-files\\2023_09_06_0007.abf"

    # Parameters
    num_points_per_segment = 24000
    sampling_rate = 2000  # 2000 Hz
    num_epochs = 10  # Specify the number of epochs you want to calculate
    cutoff_freq = 40.0  # 40 Hz cutoff frequency
    target_rec1 = 1.0  # Target recurrence rate for 1%
    target_rec5 = 5.0  # Target recurrence rate for 5%
    embedding_dims = [5, 9, 13, 17, 21, 25]  # List of embedding dimensions

    # Read and preprocess the entire channel data
    data, sampling_rate = read_abf_data(abf_file_path)
    filtered_data = low_pass_filter(data, cutoff_freq, sampling_rate)
    
    all_results = []
    for epoch_index in range(num_epochs):
        segment_start = epoch_index * num_points_per_segment
        segment_end = segment_start + num_points_per_segment
        epoch_results = process_segment(filtered_data, segment_start, segment_end, sampling_rate, embedding_dims, epoch_index, target_rec1, target_rec5, max_tau=200)
        
        all_results.extend(epoch_results)

    # Save the results to a CSV file
    save_results_to_csv(all_results, 'rqa_results.csv')

    print("Results saved to rqa_results.csv")
