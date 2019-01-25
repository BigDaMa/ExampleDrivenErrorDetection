import numpy as np
import pandas as pd
from ml.configuration.Config import Config

from ml.datasets.DataSet import DataSet


class Salaries(DataSet):

    def __init__(self):
        clean_df = pd.read_csv(Config.get("datapool.folder") + '/SALARIES/salaries_small/salaries-1_with_id.csv', header=0, dtype=object, na_filter=False)
        dirty_df = pd.read_csv(Config.get("datapool.folder") + '/SALARIES/salaries_small/dirty/dirty_salaries-1_with_id.csv', header=0, dtype=object, na_filter=False)

        super(Salaries, self).__init__("Movies", dirty_df, clean_df)





    def validate(self):
        print "validate"
