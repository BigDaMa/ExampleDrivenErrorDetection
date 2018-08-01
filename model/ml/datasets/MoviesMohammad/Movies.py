import arff
import numpy as np
import pandas as pd

from ml.datasets.DataSet import DataSet


class Movies(DataSet):

    def __init__(self):
        clean_df = pd.read_csv('/home/felix/Software/data-pool/movies/rotten_tomatoes.csv', header=0, dtype=object, na_filter=False)
        dirty_df = pd.read_csv('/home/felix/Software/data-pool/movies/dirty.csv', header=0, dtype=object, na_filter=False)

        super(Movies, self).__init__("Movies", dirty_df, clean_df)





    def validate(self):
        print "validate"
