from ml.classes.active_learning_total_uncertainty_error_correlation_class import ActiveLearningErrorCorrelation


from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.MoviesMohammad.Movies import Movies
from ml.datasets.RestaurantMohammad.Restaurant import Restaurant
from ml.datasets.BeerDataset.Beers import Beers
from ml.datasets.Citations.Citation import Citation

from ml.active_learning.classifier.XGBoostClassifier import XGBoostClassifier
import numpy as np

from ml.configuration.Config import Config
import os


path_folder = Config.get("logging.folder") + "/out/labels"
if not os.path.exists(path_folder):
    os.makedirs(path_folder)


#data_list = [FlightHoloClean, BlackOakDataSetUppercase, HospitalHoloClean, Movies, Restaurant, Beers]
data_list = [Beers]


classifier = XGBoostClassifier

parameters = []

parameters.append({'use_metadata': False, }) #unigrams

for dataset in data_list:

    '''
    dataSet,
				 classifier_model,
				 number_of_round_robin_rounds=2,
				 train_fraction=1.0,
				 ngrams=1,
				 runSVD=False,
				 is_word=False,
				 use_metadata = True,
				 use_metadata_only = False,
				 use_lstm=False,
				 user_error_probability=0.00,
				 step_size=10,
				 cross_validation_rounds=1,
				 checkN=10,
				 label_iterations=6,
				 run_round_robin=False
    '''

    method = ActiveLearningErrorCorrelation()

    data = dataset()
    fscore_lists, label = method.run(data, classifier, checkN=10)

    f_matrix = np.matrix(fscore_lists)




    import time
    ts = time.time()

    my_file = open( path_folder + '/labels_experiment_data_' + str(data.name) + "_time_" + str(ts) + '.csv', 'w+')
    my_file.write(latex)
    my_file.close()


