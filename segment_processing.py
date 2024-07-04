from knn_mutual_information import select_time_delay
from rqa_analysis import find_radii, calculate_laminarity, normalize_time_series
from pyrqa.time_series import TimeSeries

def process_segment(filtered_data, segment_start, segment_end, downsampled_rate, embedding_dims, epoch_index, target_rec1=1.0, target_rec5=5.0, max_tau=200):
    """
    Process a segment of the filtered data to calculate time delay, radii, and laminarity.

    Parameters:
    filtered_data (array-like): The filtered EEG data.
    segment_start (int): Start index of the segment.
    segment_end (int): End index of the segment.
    downsampled_rate (int): Downsampled rate in Hz.
    embedding_dims (list): List of embedding dimensions to use.
    epoch_index (int): Current epoch index.
    target_rec1 (float): Target recurrence rate for 1%.
    target_rec5 (float): Target recurrence rate for 5%.
    max_tau (int): Maximum time delay to consider.

    Returns:
    list: Results for each embedding dimension.
    """
    data_segment = filtered_data[segment_start:segment_end]
    time_delay = select_time_delay(data_segment, max_tau)
    results = []

    for embedding_dim in embedding_dims:
        print(f"\nEmbedding Dimension: {embedding_dim}")
        time_series = TimeSeries(data_segment, embedding_dimension=embedding_dim, time_delay=time_delay)
        normalize_time_series(time_series)
        rad1, rad5, average_rad = find_radii(time_series, target_rec1, target_rec5)
        laminarity = calculate_laminarity(time_series, average_rad)

        result = {
            'epoch': epoch_index,
            'embedding_dim': embedding_dim,
            'time_delay': time_delay,
            'radius_1%': rad1,
            'radius_5%': rad5,
            'average_radius': average_rad,
            'laminarity': laminarity
        }
        results.append(result)
        print(f"Epoch {epoch_index}, Embedding Dimension {embedding_dim}:")
        print(f"  1% Radius: {rad1}")
        print(f"  5% Radius: {rad5}")
        print(f"  Average Radius: {average_rad}")
        print(f"  Time Delay: {time_delay}")
        print(f"  Laminarity: {laminarity}\n")

    return results

def downsample(data, original_rate, target_rate):
    # Calculate the downsampling factor and downsample the data
    factor = original_rate // target_rate
    print(f"\nDownsampling from {original_rate} Hz to {target_rate} Hz (factor {factor})")
    return data[::factor], factor