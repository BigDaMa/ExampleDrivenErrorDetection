from ml.classes.active_learning_total_uncertainty_error_correlation_class import ActiveLearningErrorCorrelation


from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.MoviesMohammad.Movies import Movies
from ml.datasets.RestaurantMohammad.Restaurant import Restaurant

from ml.active_learning.classifier.XGBoostClassifier import XGBoostClassifier
import numpy as np


#data_list = [FlightHoloClean, BlackOakDataSetUppercase, HospitalHoloClean]
data_list =[HospitalHoloClean]


classifier = XGBoostClassifier

for dataset in data_list:
    method = ActiveLearningErrorCorrelation()

    data = dataset()
    fscore_lists, label = method.run(data, classifier, checkN=10)

    f_matrix = np.matrix(fscore_lists)

    lower_quartile = np.percentile(f_matrix, 25, axis=0)
    median = np.percentile(f_matrix, 50, axis=0)
    upper_quartile = np.percentile(f_matrix, 75, axis=0)
    minimum = np.min(f_matrix, axis=0).A1
    maximum = np.max(f_matrix, axis=0).A1

    latex = ""

    for i in range(len(lower_quartile)):
        latex += "\\boxplotlabels{"+ str(label[i]) +"}{"+ str(median[i]) +"}{"+ str(lower_quartile[i]) +"}{"+ str(upper_quartile[i]) +"}{"+ str(minimum[i]) +"}{"+ str(maximum[i]) +"}\n"

    import time
    ts = time.time()

    my_file = open('out/labels_experiment_data_' + str(data.name) + "_time_" + str(ts) + '.csv', 'w+')
    my_file.write(latex)
    my_file.close()


