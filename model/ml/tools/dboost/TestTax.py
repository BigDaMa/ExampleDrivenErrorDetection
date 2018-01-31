from ml.datasets.mohammad import MohammadDataSet
from ml.tools.dboost.TestDBoost import test

data = MohammadDataSet("tax", 20, 30, 10)

sample_size = 10
steps = 100


test(data, sample_size, steps)