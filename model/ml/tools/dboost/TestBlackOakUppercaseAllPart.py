import numpy as np

from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.tools.dboost.TestDBoost import test_multiple_sizes_gaussian

data = BlackOakDataSetUppercase()

from ml.datasets.DataColumn.DataColumn import DataColumn


error_indices = np.where(np.sum(data.matrix_is_error,axis=0)>0)[0]
print error_indices

new_data = DataColumn(data, error_indices[0])

print new_data.matrix_is_error



steps = 100
N = 5
labels = 378

nr_rows = int(float(labels) / data.shape[1])
#sizes = np.array([50, 100, 150, 200], dtype=float) # in cells
sizes = np.array([200], dtype=float)

print sizes
dirty_column_fraction = data.get_number_dirty_columns() / float(data.shape[1])
sizes /= dirty_column_fraction
sizes /= float(data.shape[1])
print sizes
row_sizes = np.array(sizes, dtype=int) # in rows
print row_sizes


log_file = "/home/felix/SequentialPatternErrorDetection/dboost/log/BlackOakUppercase_gaus_new.txt"

test_multiple_sizes_gaussian(new_data, steps, N, row_sizes, log_file)

