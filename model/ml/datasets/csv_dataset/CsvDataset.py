import numpy as np
import pandas as pd
from ml.configuration.Config import Config

from ml.datasets.DataSet import DataSet


class CsvDataset(DataSet):

    def __init__(self, clean_path, dirty_path, name="CSV_data"):
        clean_df = pd.read_csv(clean_path, header=0, dtype=object, na_filter=False)
        dirty_df = pd.read_csv(dirty_path, header=0, dtype=object, na_filter=False)

        super(CsvDataset, self).__init__(name, dirty_df, clean_df)


    def validate(self):
        print "validate"
