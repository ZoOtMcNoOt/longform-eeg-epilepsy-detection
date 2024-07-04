from knn_mutual_information import select_time_delay
from rqa_analysis import find_radii, calculate_laminarity, normalize_time_series
from pyrqa.time_series import TimeSeries

def process_segment(filtered_data, segment_start, segment_end, downsampled_rate, embedding_dims, epoch_index, target_rec1=1.0, target_rec5=5.0, max_tau=200):
    data_segment = filtered_data[segment_start:segment_end]
    time_delay = select_time_delay(data_segment, max_tau)
    results = []

    for embedding_dim in embedding_dims:
        time_series = TimeSeries(data_segment, embedding_dimension=embedding_dim, time_delay=time_delay)
        normalize_time_series(time_series)
        rad1, rad5, average_rad = find_radii(time_series, target_rec1, target_rec5)
        laminarity = calculate_laminarity(time_series, average_rad)
        
        print(f"Epoch {epoch_index + 1}, Embedding Dimension {embedding_dim}:")
        print(f"  1% Radius: {rad1}, 5% Radius: {rad5}, Average Radius: {average_rad}, Time Delay: {time_delay}, Laminarity: {laminarity}")
        
        results.append({
            "Epoch": epoch_index + 1,
            "Embedding Dimension": embedding_dim,
            "1% Radius": rad1,
            "5% Radius": rad5,
            "Average Radius": average_rad,
            "Time Delay": time_delay,
            "Laminarity": laminarity
        })

    return results

def downsample(data, original_rate, target_rate):
    factor = original_rate // target_rate
    return data[::factor], factor
