import numpy as np
import pandas as pd

from ml.datasets.DataSet import DataSet


class Simulator(DataSet):
	name = "Simulator"

	def __init__(self, n):
		folder = "/tmp/my_data/"
		dirty_pd = pd.read_csv(folder + "dirty_data" + str(n) + ".csv", header=0, dtype=object, na_filter=False)
		clean_pd = pd.read_csv(folder + "clean_data" + str(n) + ".csv", header=0, dtype=object, na_filter=False)

		super(Simulator, self).__init__(Simulator.name, dirty_pd, clean_pd)

	def validate(self):
		print "validate"
