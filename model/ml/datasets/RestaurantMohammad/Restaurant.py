import numpy as np
import pandas as pd
from ml.configuration.Config import Config

from ml.datasets.DataSet import DataSet


class Restaurant(DataSet):
    name = "Restaurant"
    def __init__(self):
        clean_df = pd.read_csv(Config.get("datapool.folder") + '/restaurants/yellow_pages.csv', header=0, dtype=object, na_filter=False)
        dirty_df = pd.read_csv(Config.get("datapool.folder") + '/restaurants/dirty.csv', header=0, dtype=object, na_filter=False)

        super(Restaurant, self).__init__("Restaurant", dirty_df, clean_df)





    def validate(self):
        print "validate"
