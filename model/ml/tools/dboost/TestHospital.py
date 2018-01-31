from ml.datasets.hospital import HospitalHoloClean
from ml.tools.dboost.TestDBoost import test

data = HospitalHoloClean()

sample_size = 10
steps = 100

test(data, sample_size, steps)