import pyabf
from segment_processing import process_segment, downsample
from save_results import save_results_to_csv

def main():
    # Path to the ABF file
    abf_file_path = "C:\\Users\\grant\\OneDrive - Texas A&M University\\D2 1mm TBI control\\abf-files\\2023_09_06_0007.abf"
    
    # Read the ABF file
    abf = pyabf.ABF(abf_file_path)
    original_sampling_rate = abf.dataRate
    target_sampling_rate = 400  # Target downsampling rate in Hz
    epoch_duration_seconds = 12  # Duration of each epoch in seconds
    embedding_dims = [5, 9, 13, 17, 21, 25]
    
    results = []

    # Process each channel in the ABF file
    for channel in range(abf.channelCount):
        abf.setSweep(0, channel=channel)
        channel_data = abf.sweepY
        
        # Downsample the data
        downsampled_data, _ = downsample(channel_data, original_sampling_rate, target_sampling_rate)
        epoch_samples = int(epoch_duration_seconds * target_sampling_rate)
        
        # Process each epoch
        total_epochs = len(downsampled_data) // epoch_samples
        for epoch_index in range(total_epochs):
            segment_start = epoch_index * epoch_samples
            segment_end = segment_start + epoch_samples
            if segment_end > len(downsampled_data):
                break
            
            print(f"\nProcessing Channel {channel + 1}, Epoch {epoch_index + 1}")
            epoch_results = process_segment(downsampled_data, segment_start, segment_end, target_sampling_rate, embedding_dims, epoch_index)
            results.extend(epoch_results)

    # Save the results to a CSV file
    save_results_to_csv(results, 'rqa_results.csv')
    print("Results saved to rqa_results.csv")

if __name__ == "__main__":
    main()
