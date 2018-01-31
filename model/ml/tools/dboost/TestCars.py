from ml.datasets.mohammad import MohammadDataSet
from ml.tools.dboost.TestDBoost import run_params_gaussian

data = MohammadDataSet("cars", 30, 20, 20)

sample_size = 10
steps = 100


best_params = {}
best_params['gaussian'] = 1.0
best_params['statistical'] = 0.5
run_params_gaussian(data, best_params)