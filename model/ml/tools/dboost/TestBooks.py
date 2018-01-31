import sys

from ml.datasets.mohammad import MohammadDataSet
from ml.tools.dboost.TestDBoost import test

reload(sys)
sys.setdefaultencoding('utf-8')

data = MohammadDataSet("books", 30, 30, 10)

sample_size = 10
steps = 100


test(data, sample_size, steps)