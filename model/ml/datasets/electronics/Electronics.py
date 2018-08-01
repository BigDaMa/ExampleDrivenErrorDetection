import numpy as np
import pandas as pd

from ml.configuration.Config import Config
from ml.datasets.DataSet import DataSet


class Electronics(DataSet):
	name = "electronics"

	def select_columns(self, data):
		return data[["Brand","Name","Price","Features"]]

	def __init__(self):
		amazon = "/home/felix/datasets/duplicate_data/electronics/csv_files/amazon.csv"
		bestbuy = "/home/felix/datasets/duplicate_data/electronics/csv_files/best_buy.csv"
		labelled = "/home/felix/datasets/duplicate_data/electronics/csv_files/labeled_data.csv"

		amazon_df = self.select_columns(pd.read_csv(amazon, header=0, dtype=object, na_filter=False, names=["ID","Brand","Name","Amazon_Price","Price","Features"]))
		bestbuy_df = self.select_columns(pd.read_csv(bestbuy, header=0, dtype=object, na_filter=False, names=["ID","Brand","Name","Price","Description","Features"]))
		l = pd.read_csv(labelled, header=5, dtype=object, na_filter=False)


		print bestbuy_df

		amazon_clean = amazon_df.copy()

		#left = amazon

		bestbuy_ids = l.values[:, 2]
		amazon_ids = l.values[:, 1]
		is_duplicate = l.values[:,3]

		for t in range(len(bestbuy_ids)):
			if is_duplicate[t] == '1':
				print "Before:" + str(amazon_clean.values[int(amazon_ids[t]) - 1, :])

				for col in amazon_clean.columns:
					amazon_clean.at[int(amazon_ids[t]) - 1, col] = bestbuy_df[col].values[int(bestbuy_ids[t]) - 1]

				print "After:" + str(amazon_clean.values[int(amazon_ids[t]) - 1, :])

		dirty_pd = amazon_df.append(bestbuy_df)
		clean_pd = amazon_clean.append(bestbuy_df)

		super(Electronics, self).__init__(Electronics.name, dirty_pd, clean_pd)




	def validate(self):
		print "validate"

if __name__ == '__main__':
	data = Electronics()

	print np.sum(data.matrix_is_error, axis=0) / float(data.shape[0])

	print data.shape

	import csv
	#data.clean_pd.to_csv('/tmp/electronics_clean.csv', index=False, quoting=csv.QUOTE_ALL)
	#data.dirty_pd.to_csv('/tmp/electronics_dirty.csv', index=False, quoting=csv.QUOTE_ALL)