from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import MaximumMetric
from pyrqa.computation import RQAComputation
import numpy as np

def calculate_recurrence_rate(time_series, radius):
    settings = Settings(time_series, analysis_type=Classic, neighbourhood=FixedRadius(radius), similarity_measure=MaximumMetric, theiler_corrector=1)
    rqa = RQAComputation.create(settings)
    results = rqa.run()
    results.min_diagonal_line_length = 5
    recurrence_rate = results.recurrence_rate * 100  # Convert to percentage
    return recurrence_rate

def binary_search_radius(time_series, target_recurrence, tolerance=0.01, max_iter=100):
    low, high = 0.0, 1.0  # Assuming the radius is between 0 and 1 initially
    iteration = 0

    while low <= high and iteration < max_iter:
        mid = (low + high) / 2
        rec_rate = calculate_recurrence_rate(time_series, mid)
        print(f"Radius: {mid}, Recurrence Rate: {rec_rate}%")

        if abs(rec_rate - target_recurrence) <= tolerance:
            return mid
        
        if rec_rate < target_recurrence:
            low = mid + tolerance / 10  # Adjust increment for precision
        else:
            high = mid - tolerance / 10  # Adjust decrement for precision

        iteration += 1

    return mid

def find_radii(time_series, target_rec1, target_rec5):
    rad1 = binary_search_radius(time_series, target_rec1)
    rad5 = binary_search_radius(time_series, target_rec5)
    average_rad = (rad1 + rad5) / 2
    return rad1, rad5, average_rad

def calculate_laminarity(time_series, radius):
    settings = Settings(time_series, analysis_type=Classic, neighbourhood=FixedRadius(radius), similarity_measure=MaximumMetric, theiler_corrector=1)
    rqa = RQAComputation.create(settings)
    results = rqa.run()
    results.min_diagonal_line_length = 5
    laminarity = results.laminarity
    return laminarity
