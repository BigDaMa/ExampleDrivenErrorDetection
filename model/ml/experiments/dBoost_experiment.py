from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.MoviesMohammad.Movies import Movies
from ml.datasets.RestaurantMohammad.Restaurant import Restaurant

import time
from ml.tools.dboost.TestDBoost import test_multiple_sizes_hist
from ml.tools.dboost.TestDBoost import test_multiple_sizes_gaussian
from ml.tools.dboost.TestDBoost import test_multiple_sizes_mixture


import numpy as np


data_list =[FlightHoloClean, BlackOakDataSetUppercase, HospitalHoloClean, Restaurant, Movies]

steps = 100 #size of grid
N = 1

dBoost_methods = [test_multiple_sizes_hist, test_multiple_sizes_gaussian, test_multiple_sizes_mixture]


for dataset in data_list:
    data = dataset()
    number_rows = data.shape[0] #how big is the evaluation set
    for dBoost in dBoost_methods:
        ts = time.time()
        log_file = "out/dboost/" + str(data.name) + "_time_" + str(ts) + "_dBoost_" + dBoost.func_name  + ".txt"
        dBoost(data, steps, N, [number_rows], log_file)

'''
str(best_params) + ", " +
str(runtime) + ", " +
str(precision) + ", " +
str(recall) + ", " +
str(best_fscore_all)
'''
