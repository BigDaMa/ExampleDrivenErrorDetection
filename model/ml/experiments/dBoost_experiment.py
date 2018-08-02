from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.MoviesMohammad.Movies import Movies
from ml.datasets.RestaurantMohammad.Restaurant import Restaurant

import time
from ml.tools.dboost.TestDBoost import test_multiple_sizes_hist
from ml.tools.dboost.TestDBoost import test_multiple_sizes_gaussian
from ml.tools.dboost.TestDBoost import test_multiple_sizes_mixture

from ml.configuration.Config import Config
import numpy as np
import os


path_folder = Config.get("logging.folder") + "/out/dboost"

if not os.path.exists(path_folder):
    os.makedirs(path_folder)


data_list = [FlightHoloClean, BlackOakDataSetUppercase, HospitalHoloClean, Restaurant, Movies]

steps = 100 #size of grid
N = 1

dBoost_methods = [test_multiple_sizes_hist, test_multiple_sizes_gaussian, test_multiple_sizes_mixture]

for dataset in data_list:
    data = dataset()

    for dBoost in dBoost_methods:
        ts = time.time()
        log_file = path_folder + "/" + str(data.name) + "_time_" + str(ts) + "_dBoost_" + dBoost.func_name  + ".txt"
        dBoost(data, steps, N, [data.shape[0]], log_file)

'''
str(best_params) + ", " +
str(runtime) + ", " +
str(precision) + ", " +
str(recall) + ", " +
str(best_fscore_all)
'''
