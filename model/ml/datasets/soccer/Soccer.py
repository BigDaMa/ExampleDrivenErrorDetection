from ml.configuration.Config import Config

from ml.datasets.csv_dataset.CsvDataset import CsvDataset

class Soccer(CsvDataset):

    def __init__(self):
        super(Soccer, self).__init__(Config.get("datapool.folder") + '/Geerts/soccer/clean/Soccer.csv', Config.get("datapool.folder") + '/Geerts/soccer/dirty3/Soccer_dirty_10pct_1cfd_3.csv', 'Soccer')


    def validate(self):
        print "validate"