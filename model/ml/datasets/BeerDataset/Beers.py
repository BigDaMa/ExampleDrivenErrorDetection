import numpy as np
import pandas as pd
from ml.configuration.Config import Config

from ml.datasets.DataSet import DataSet


class Beers(DataSet):

    def __init__(self):
        clean_df = pd.read_csv(Config.get("datapool.folder") + '/BEERS/beers-and-breweries.csv', header=0, dtype=object, na_filter=False)
        dirty_df = pd.read_csv(Config.get("datapool.folder") + '/BEERS/dirty-beers-and-breweries.csv', header=0, dtype=object, na_filter=False)

        super(Beers, self).__init__("BEERS", dirty_df, clean_df)


    def validate(self):
        print "validate"
