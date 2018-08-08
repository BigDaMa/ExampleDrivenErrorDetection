import numpy as np
import pandas as pd
from ml.configuration.Config import Config

from ml.datasets.DataSet import DataSet


class Citation(DataSet):

    def __init__(self):
        clean_df = pd.read_csv(Config.get("datapool.folder") + '/Citations/citation.csv', header=0, dtype=object, na_filter=False, encoding="utf8")
        dirty_df = pd.read_csv(Config.get("datapool.folder") + '/Citations/dirty.csv', header=0, dtype=object, na_filter=False, encoding="utf8")

        super(Citation, self).__init__("Citation", dirty_df, clean_df)





    def validate(self):
        print "validate"
