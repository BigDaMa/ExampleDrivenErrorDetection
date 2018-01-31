import pandas as pd

from ml.datasets.DataSet import DataSet


class FlightLarysa(DataSet):
    def __init__(self):
        path_to_dirty = "/home/felix/SequentialPatternErrorDetection/flight/larysa/flights-dirty.csv"
        path_to_clean = "/home/felix/SequentialPatternErrorDetection/flight/larysa/flights-ground-truth.csv"

        dirty_pd = pd.read_csv(path_to_dirty, header=0, dtype=object)
        clean_pd = pd.read_csv(path_to_clean, header=0, dtype=object)

        super(FlightLarysa, self).__init__("Flight Larysa", dirty_pd, clean_pd)

    def validate(self):
        print "validate"
