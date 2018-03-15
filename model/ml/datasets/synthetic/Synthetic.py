import numpy as np
import pandas as pd

from ml.datasets.DataSet import DataSet


class Synthetic(DataSet):
	name = "Synthetic"

	def __init__(self, rows, datasets, columns, error_fraction, error_types, seed):
		np.random.seed(seed=seed)

		assert len(datasets) == len(columns)
		assert len(error_fraction) == len(columns)
		assert len(error_fraction) == len(error_types)

		all_column_data = []
		column_names = []

		for c in range(len(columns)):
			all_indices = range(datasets[c].shape[0])
			np.random.shuffle(all_indices)
			all_column_data.append(pd.Series(datasets[c].clean_pd.values[all_indices[0:rows], columns[c]]))
			column_names.append(datasets[c].clean_pd.columns[c] + "_" + str(c))

		#create clean pd from "rows, datasets, columns"
		clean_pd = pd.concat(all_column_data, axis=1)

		print clean_pd

		# create dirty pd by error_fraction, error_types
		dirty_pd = clean_pd.copy()

		for c in range(dirty_pd.shape[1]):
			dirty_pd[dirty_pd.columns[c]] = error_types[c].error(dirty_pd[dirty_pd.columns[c]].values, error_fraction[c])


		print dirty_pd

		super(Synthetic, self).__init__(Synthetic.name, dirty_pd, clean_pd)

	def validate(self):
		print "validate"
