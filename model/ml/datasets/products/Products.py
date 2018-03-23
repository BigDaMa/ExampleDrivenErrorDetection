import numpy as np
import pandas as pd

from ml.configuration.Config import Config
from ml.datasets.DataSet import DataSet


class Products(DataSet):
	name = "products"

	def select_columns(self, data):
		return data[['title', 'brand', 'price', 'shortdescr', 'longdescr', 'dimensions', 'shipweight']]

	def __init__(self):
		amazon = "/home/felix/new_datasets/products/amazon.csv"
		wallmart = "/home/felix/new_datasets/products/walmart.csv"
		labelled = "/home/felix/new_datasets/products/matches_walmart_amazon.csv"

		amazon_df = self.select_columns(pd.read_csv(amazon, header=0, dtype=object, na_filter=False, names=["custom_id","url","asin","brand","modelno","category1","pcategory1","category2","pcategory2","title","listprice","price","prodfeatures","techdetails","shortdescr","longdescr","dimensions","imageurl","itemweight","shipweight","orig_prodfeatures","orig_techdetails"]))
		wallmart_df = self.select_columns(pd.read_csv(wallmart, header=0, dtype=object, na_filter=False, names=["custom_id","id","upc","brand","groupname","title","price","shelfdescr","shortdescr","longdescr","imageurl","orig_shelfdescr","orig_shortdescr","orig_longdescr","modelno","shipweight","dimensions"]))
		l = pd.read_csv(labelled, header=0, dtype=object, na_filter=False)

		amazon_clean = amazon_df.copy()

		#left = wallmart

		wallmart_ids = l.values[:, 0]
		amazon_ids = l.values[:, 1]

		for t in range(len(wallmart_ids)):
			print "Before:" + str(amazon_clean.values[int(amazon_ids[t]) - 1, :])

			for col in amazon_clean.columns:
				amazon_clean.at[int(amazon_ids[t]) - 1, col] = wallmart_df[col].values[int(wallmart_ids[t]) - 1]

			print "After:" + str(amazon_clean.values[int(amazon_ids[t]) - 1, :])

		dirty_pd = amazon_df.append(wallmart_df)
		clean_pd = amazon_clean.append(wallmart_df)

		super(Products, self).__init__(Products.name, dirty_pd, clean_pd)




	def validate(self):
		print "validate"

if __name__ == '__main__':
	data = Products()

	print np.sum(data.matrix_is_error, axis=0) / float(data.shape[0])

	print data.shape

	import csv
	#data.clean_pd.to_csv('/tmp/products_clean.csv', index=False, quoting=csv.QUOTE_ALL)
	#data.dirty_pd.to_csv('/tmp/products_dirty.csv', index=False, quoting=csv.QUOTE_ALL)