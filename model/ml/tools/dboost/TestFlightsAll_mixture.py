import numpy as np

from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.tools.dboost.TestDBoost import test_multiple_sizes_mixture

data = FlightHoloClean()

'''
steps = 100
N = 5
sizes = [10, 20, 30, 40, 50]

test_multiple_sizes_hist(data, steps, N, sizes)
'''

steps = 100
N = 10
labels = 216

nr_rows = int(float(labels) / data.shape[1])
sizes = np.array([50, 100, 150, 200], dtype=float) # in cells
#sizes = np.array([200], dtype=float) # in cells

print sizes
dirty_column_fraction = data.get_number_dirty_columns() / float(data.shape[1])
sizes /= dirty_column_fraction
sizes /= float(data.shape[1])
print sizes
row_sizes = np.array(sizes, dtype=int) # in rows

log_file = "/home/felix/ExampleDrivenErrorDetection/log/dBoost/Flights_mixture_new.txt"

test_multiple_sizes_mixture(data, steps, N, row_sizes, log_file)