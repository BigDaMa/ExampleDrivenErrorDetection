import numpy as np
import pandas as pd
from ml.configuration.Config import Config

from ml.datasets.DataSet import DataSet


class Beers(DataSet):

    def __init__(self):
        clean_df = pd.read_csv('/home/felix/data_more/Beers_Mohammad/clean.csv', header=0, dtype=object, na_filter=False)
        dirty_df = pd.read_csv('/home/felix/data_more/Beers_Mohammad/dirty.csv', header=0, dtype=object, na_filter=False)

        super(Beers, self).__init__("Beers", dirty_df, clean_df)





    def validate(self):
        print "validate"
