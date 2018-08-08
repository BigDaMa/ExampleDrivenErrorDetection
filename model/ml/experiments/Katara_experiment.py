from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.MoviesMohammad.Movies import Movies
from ml.datasets.RestaurantMohammad.Restaurant import Restaurant
from ml.datasets.BeerDataset.Beers import Beers
from ml.datasets.Citations.Citation import Citation

import time

from ml.configuration.Config import Config
import numpy as np
import os
from ml.tools.katara_new.Katara import Katara


path_folder = Config.get("logging.folder") + "/out/katara"

if not os.path.exists(path_folder):
    os.makedirs(path_folder)

#data_list = [FlightHoloClean, BlackOakDataSetUppercase, HospitalHoloClean, Movies, Restaurant, Beers]
data_list = [Movies]

for dataset in data_list:

    data = dataset()

    ts = time.time()
    log_file = path_folder + "/" + str(data.name) + "_time_" + str(ts) + "_KATARA_"  + ".txt"

    dirty_dataset = '/tmp/dirty_dataset.csv'
    data.dirty_pd.to_csv(dirty_dataset, index=False, encoding='utf-8')

    start_time = time.time()

    command = "cd " + Config.get("abstractionlayer.folder") + "/\n" + "python2 cleaning_api.py"
    print command
    os.system(command)

    if os.stat('/tmp/katara_log_felix.txt').st_size == 0:
        print "KATARA did not work"
    else:
        tool = Katara('/tmp/katara_log_felix.txt', data)

        my_file = open(log_file, 'w+')
        my_file.write("Fscore: " + str(tool.calculate_total_fscore()) + "\n")
        my_file.write("Precision: " + str(tool.calculate_total_precision()) + "\n")
        my_file.write("Recall: " + str(tool.calculate_total_recall()) + "\n")
        my_file.write("Runtime: " + str(time.time() - start_time) + "\n")
        my_file.close()