import pandas as pd
from ml.datasets.DataSet import DataSet
from ml.configuration.Config import Config


class BartDataset(DataSet):
	name = "BartDataset"

	def __init__(self, dataset, folder):
		path_to_dirty = "/home/felix/ExampleDrivenErrorDetection/bart_data/" + folder + "/dirty_person.csv"
		dirty_pd_init = pd.read_csv(path_to_dirty, header=0, dtype=object, na_filter=False)

		dirty_pd = dirty_pd_init

		clean_pd = dataset.clean_pd

		super(BartDataset, self).__init__(BartDataset.name, dirty_pd, clean_pd)


def validate(self):
	print "validate"


if __name__ == '__main__':
	from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
	import numpy as np

	data = BartDataset(BlackOakDataSetUppercase(), "CityFD_10percent_Remove")

	'''
	from ml.datasets.salary_data.Salary import Salary

	#outlier data
	datan = Salary()
	def convert_to_int(value):
		return str(int(float(value)))
	datan.clean_pd[datan.clean_pd.columns[8]] = datan.clean_pd[datan.clean_pd.columns[8]].apply(convert_to_int)
	data = BartDataset(datan, "Salary_outlier_5percent")
	'''

	error_fractions = np.sum(data.matrix_is_error, axis=0)


	print data.clean_pd.columns
	print error_fractions
	print error_fractions / float(data.shape[0])
