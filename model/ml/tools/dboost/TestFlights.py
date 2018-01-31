from ml.datasets.flights import FlightHoloClean
from ml.tools.dboost.TestDBoost import test

data = FlightHoloClean()

sample_size = 10
steps = 100


test(data, sample_size, steps)