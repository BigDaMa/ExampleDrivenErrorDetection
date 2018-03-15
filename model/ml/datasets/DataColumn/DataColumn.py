import numpy as np
import pandas as pd

from ml.datasets.DataSet import DataSet


class DataColumn(DataSet):
	name = "Column"

	def __init__(self, dataset, column_id):
		all_column_data = []
		column_names = []

		clean_pd = pd.DataFrame(dataset.clean_pd[dataset.clean_pd.columns[column_id]])
		dirty_pd = pd.DataFrame(dataset.dirty_pd[dataset.dirty_pd.columns[column_id]])

		print clean_pd

		super(DataColumn, self).__init__(DataColumn.name, dirty_pd, clean_pd)

	def validate(self):
		print "validate"
