from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.MoviesMohammad.Movies import Movies
from ml.datasets.RestaurantMohammad.Restaurant import Restaurant
from ml.datasets.BeersMohammad.Beers import Beers
from ml.datasets.Citations.Citation import Citation
from ml.datasets.salary_data.Salary import Salary

from ml.tools.dboost.TestDBoost import run_gaussian_stat
from ml.tools.dboost.TestDBoost import run_histogram_stat
from ml.tools.dboost.TestDBoost import run_histogram_mixture

import time
import numpy as np
import glob
from ml.configuration.Config import Config
import os

mypath = Config.get("logging.folder") + "/out/server_dboost"
mylist = [f for f in glob.glob(mypath + "/*.txt")]

datasets = [FlightHoloClean(),
            Beers(),
            BlackOakDataSetUppercase(),
            HospitalHoloClean(),
            Movies(),
            Restaurant(),
            Citation(),
            Salary()]


N = 10

path_folder = Config.get("logging.folder") + "/out/dboost_runtime"

if not os.path.exists(path_folder):
    os.makedirs(path_folder)

log_file = open(
        path_folder + '/dboost_runtime'+ str(time.time()) + '.csv', 'w+')

for file_name in mylist:
    data_name = file_name.split('/')[-1].split('.')[0].split('_')[0]
    dataset = None
    for d in range(len(datasets)):
        if datasets[d].name == data_name:
            dataset = datasets[d]

    sample_file = "/tmp/data_sample.csv"

    dataset.dirty_pd.to_csv(sample_file, index=False, encoding="utf8")

    print file_name

    method = None
    parameter_dict = {}
    parameter_dict['sample_file'] = sample_file

    with open(file_name) as config_file:
        counter = 0
        for line in config_file:
            if counter == 2:
                data = line.replace(':', ',').replace('}', ',').split(',')

                if 'dBoost_test_multiple_sizes_mixture.txt' in file_name:
                    parameter_dict['threshold'] = float(data[1])
                    parameter_dict['n_subpops'] = float(data[3])
                    parameter_dict['statistical'] = float(data[5])
                    method = run_histogram_mixture


                if 'dBoost_test_multiple_sizes_gaussian.txt' in file_name:
                    parameter_dict['gaussian'] = float(data[1])
                    parameter_dict['statistical'] = float(data[3])
                    method = run_gaussian_stat

                if 'dBoost_test_multiple_sizes_hist.txt' in file_name:
                    parameter_dict['peak'] = float(data[1])
                    parameter_dict['statistical'] = float(data[3])
                    parameter_dict['outlier'] = float(data[5])
                    method = run_histogram_stat

                break
            counter += 1

    runtimes = []
    for i_run in range(N):
        ts = time.time()
        method(**parameter_dict)
        runtime = time.time() - ts
        runtimes.append(runtime)
    log_file.write(file_name + ": " + str(np.mean(runtimes)) + '\n\n')

log_file.close()









