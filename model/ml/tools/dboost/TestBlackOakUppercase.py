from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.tools.dboost.TestDBoost import test

data = BlackOakDataSetUppercase()

sample_size = 10
steps = 100


test(data, sample_size, steps)