import numpy as np
import pandas as pd
from ml.configuration.Config import Config

from ml.datasets.DataSet import DataSet

class Beers(DataSet):
    name = "Beers"

    def __init__(self):
        clean_df = pd.read_csv(Config.get("datapool.folder") + '/Beers_Mohammad/clean.csv', header=0, dtype=object, na_filter=False)
        dirty_df = pd.read_csv(Config.get("datapool.folder") + '/Beers_Mohammad/dirty.csv', header=0, dtype=object, na_filter=False)

        clean_df = clean_df.drop('ounces', 1)
        dirty_df = dirty_df.drop('ounces', 1)

        super(Beers, self).__init__("Beers", dirty_df, clean_df)





    def validate(self):
        print "validate"


if __name__ == '__main__':
    data = Beers()

    print data.clean_pd.columns
    print list(np.sum(data.matrix_is_error, axis=0) / float(data.shape[0]))