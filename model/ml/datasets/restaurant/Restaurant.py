import arff
import numpy as np
import pandas as pd

from ml.datasets.DataSet import DataSet


class Restaurant(DataSet):

    def __init__(self):
        path_to_dirty = "/home/felix/SequentialPatternErrorDetection/switch_restaurant/restaurant/fz-nophone.arff"
        #path_to_clean = "/home/felix/SequentialPatternErrorDetection/holoclean_data/hospital_clean.csv"

        dirty_wrong_format = np.array(arff.load(open(path_to_dirty, 'rb'))['data'])
        #clean_wrong_format = pd.read_csv(path_to_clean, header=0, dtype=object)

        columns = ['name', 'addr', 'city', 'type', 'class']

        dirty_pd = pd.DataFrame(dirty_wrong_format,columns=columns)

        print dirty_pd

        dirty_pd = dirty_pd.drop(['class'], 1)

        super(Restaurant, self).__init__("Restaurant", dirty_pd, None)





    def validate(self):
        print "validate"
