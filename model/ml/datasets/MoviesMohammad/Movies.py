import numpy as np
import pandas as pd
from ml.configuration.Config import Config

from ml.datasets.DataSet import DataSet


class Movies(DataSet):
    name = "Movies"
    def __init__(self):
        clean_df = pd.read_csv(Config.get("datapool.folder") + '/movies/rotten_tomatoes.csv', header=0, dtype=object, na_filter=False)
        dirty_df = pd.read_csv(Config.get("datapool.folder") + '/movies/dirty.csv', header=0, dtype=object, na_filter=False)

        super(Movies, self).__init__("Movies", dirty_df, clean_df)





    def validate(self):
        print "validate"
