from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
import numpy as np
from scipy.spatial.distance import pdist

def calculate_recurrence_rate(time_series, radius):
    settings = Settings(time_series, analysis_type=Classic, neighbourhood=FixedRadius(radius), similarity_measure=EuclideanMetric, theiler_corrector=1)
    rqa = RQAComputation.create(settings)
    results = rqa.run()
    recurrence_rate = results.recurrence_rate * 100  # Convert to percentage
    return recurrence_rate

def binary_search_radius(time_series, target_recurrence, max_distance, tolerance=0.01, max_iter=100):
    low, high = 0.0, max_distance
    best_radius, best_diff = low, float('inf')

    for _ in range(max_iter):
        mid = (low + high) / 2
        rec_rate = calculate_recurrence_rate(time_series, mid)
        diff = abs(rec_rate - target_recurrence)

        if diff < best_diff:
            best_diff, best_radius = diff, mid

        if diff <= tolerance:
            break

        if rec_rate < target_recurrence:
            low = mid + tolerance
        else:
            high = mid - tolerance

    return best_radius

def find_radii(time_series, target_rec1, target_rec5):
    max_distance = calculate_max_distance(time_series)
    rad1 = binary_search_radius(time_series, target_rec1, max_distance)
    rad5 = binary_search_radius(time_series, target_rec5, max_distance)
    average_rad = (rad1 + rad5) / 2
    return rad1, rad5, average_rad

def normalize_time_series(time_series):
    data = np.array(time_series.data)
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    time_series.data = normalized_data

def calculate_max_distance(time_series):
    data = np.array(time_series.data).reshape(-1, 1)
    return np.max(pdist(data, 'euclidean'))

def calculate_laminarity(time_series, radius):
    settings = Settings(time_series, analysis_type=Classic, neighbourhood=FixedRadius(radius), similarity_measure=EuclideanMetric, theiler_corrector=1)
    rqa = RQAComputation.create(settings)
    return rqa.run().laminarity
