from ml.configuration.Config import Config

from ml.datasets.csv_dataset.CsvDataset import CsvDataset

class Adult(CsvDataset):

    def __init__(self):
        super(Adult, self).__init__(Config.get("datapool.folder") + '/Geerts/adult/clean/adult.csv', Config.get("datapool.folder") + '/Geerts/adult/dirty1/adult_dirty_10pct_1cfd_1.csv', 'Adult')


    def validate(self):
        print "validate"
