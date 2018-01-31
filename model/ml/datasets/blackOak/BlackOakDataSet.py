import pandas as pd
from ml.configuration.Config import Config
from ml.datasets.DataSet import DataSet


class BlackOakDataSet(DataSet):
    def __init__(self):
        path_to_dirty = Config.get("blackoak.data") + "/inputDB.csv"
        path_to_clean = Config.get("blackoak.data") + "/groundDB.csv"

        dirty_pd = pd.read_csv(path_to_dirty, header=0, dtype=object)
        clean_pd = pd.read_csv(path_to_clean, header=0, dtype=object)

        super(BlackOakDataSet, self).__init__("BlackOak", dirty_pd, clean_pd)

    def validate(self):
        print "validate"
