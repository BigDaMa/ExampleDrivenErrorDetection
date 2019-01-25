import numpy as np

from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.tools.dboost.TestDBoost import test_multiple_sizes_gaussian
from ml.tools.dboost.TestDBoost import toLatex
from ml.configuration.Config import Config
import os
import time

data = FlightHoloClean()

steps = 100 #grid for search
N = 1#10 # number runs


defined_range_labeled_cells = [100]#[20,40,60,80,100,120]

sizes = np.array(defined_range_labeled_cells, dtype=float) # in cells

print sizes
dirty_column_fraction = data.get_number_dirty_columns() / float(data.shape[1])
sizes /= dirty_column_fraction #cells converted
sizes /= float(data.shape[1]) #cells to rows
row_sizes = np.array(sizes, dtype=int) # in rows

path_folder = Config.get("logging.folder") + "/out/dboost"
log_file = path_folder + "/Flights_gaus_new " + time.time() + ".txt"

if not os.path.exists(path_folder):
    os.makedirs(path_folder)


avg_times, avg_fscores, avg_precision, avg_recall, std_fscores, std_precision, std_recall = test_multiple_sizes_gaussian(data, steps, N, row_sizes, log_file)

toLatex(defined_range_labeled_cells, avg_times, avg_fscores, avg_precision, avg_recall, std_fscores, std_precision, std_recall, log_file)







