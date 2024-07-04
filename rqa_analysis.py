from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
import numpy as np
from scipy.spatial.distance import pdist

def calculate_recurrence_rate(time_series, radius):
    """
    Calculate the recurrence rate for a given radius.

    Parameters:
    time_series (TimeSeries): The time series data for RQA.
    radius (float): Radius for the recurrence neighborhood.

    Returns:
    float: Recurrence rate as a percentage.
    """
    settings = Settings(time_series, analysis_type=Classic, neighbourhood=FixedRadius(radius), similarity_measure=EuclideanMetric, theiler_corrector=1)
    rqa = RQAComputation.create(settings)
    results = rqa.run()
    results.min_diagonal_line_length = 5
    recurrence_rate = results.recurrence_rate * 100  # Convert to percentage
    return recurrence_rate

def binary_search_radius(time_series, target_recurrence, max_distance, tolerance=0.01, max_iter=100):
    """
    Perform a binary search to find the radius that achieves the target recurrence rate.

    Parameters:
    time_series (TimeSeries): The time series data for RQA.
    target_recurrence (float): Target recurrence rate.
    max_distance (float): Maximum distance in the time series.
    tolerance (float): Tolerance for the recurrence rate.
    max_iter (int): Maximum number of iterations for the binary search.

    Returns:
    float: Radius that achieves the target recurrence rate.
    """
    low, high = 0.0, max_distance
    best_radius, best_diff = low, float('inf')

    print("\nBinary search for radius:")
    for i in range(max_iter):
        mid = (low + high) / 2
        rec_rate = calculate_recurrence_rate(time_series, mid)
        diff = abs(rec_rate - target_recurrence)
        print(f"  Iteration {i + 1}: Radius = {mid:.6f}, Recurrence Rate = {rec_rate:.6f}%, Difference = {diff:.6f}")

        if diff < best_diff:
            best_diff, best_radius = diff, mid

        if diff <= tolerance:
            print("  Tolerance level reached\n")
            break

        if rec_rate < target_recurrence:
            low = mid + tolerance
        else:
            high = mid - tolerance

    print(f"  Best radius found: {best_radius} with difference: {best_diff}\n")
    return best_radius

def find_radii(time_series, target_rec1, target_rec5):
    """
    Find the radii corresponding to 1% and 5% recurrence rates.

    Parameters:
    time_series (TimeSeries): The time series data for RQA.
    target_rec1 (float): Target recurrence rate for 1%.
    target_rec5 (float): Target recurrence rate for 5%.

    Returns:
    tuple: Radii for 1%, 5% recurrence rates, and their average.
    """
    max_distance = calculate_max_distance(time_series)
    print(f"\nMax distance in time series: {max_distance}\n")
    rad1 = binary_search_radius(time_series, target_rec1, max_distance)
    rad5 = binary_search_radius(time_series, target_rec5, max_distance)
    average_rad = (rad1 + rad5) / 2
    return rad1, rad5, average_rad

def normalize_time_series(time_series):
    """
    Normalize the time series data to the range [0, 1].

    Parameters:
    time_series (TimeSeries): The time series data for RQA.
    """
    data = np.array(time_series.data)
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    time_series.data = normalized_data
    print(f"\nTime series normalized\n")

def calculate_max_distance(time_series):
    """
    Calculate the maximum distance in the time series data.

    Parameters:
    time_series (TimeSeries): The time series data for RQA.

    Returns:
    float: Maximum distance in the time series data.
    """
    data = np.array(time_series.data).reshape(-1, 1)
    return np.max(pdist(data, 'euclidean'))

def calculate_laminarity(time_series, radius):
    """
    Calculate the laminarity for a given radius.

    Parameters:
    time_series (TimeSeries): The time series data for RQA.
    radius (float): Radius for the recurrence neighborhood.

    Returns:
    float: Laminarity as a percentage.
    """
    settings = Settings(time_series, analysis_type=Classic, neighbourhood=FixedRadius(radius), similarity_measure=EuclideanMetric, theiler_corrector=1)
    rqa = RQAComputation.create(settings)
    rqa.min_diagonal_line_length = 5
    return rqa.run().laminarity * 100  # Convert to percentage
