import numpy as np
import pandas as pd

from ml.configuration.Config import Config
from ml.datasets.DataSet import DataSet


class EBooks1(DataSet):
	name = "ebooks1"

	def select_columns(self, data):
		return data[['title', 'author', 'date', 'publisher', 'price', 'length', 'description']]

	def __init__(self):
		ebooks = "/home/felix/new_datasets/ebooks1/csv_files/ebooks.csv"
		itunes = "/home/felix/new_datasets/ebooks1/csv_files/itunes.csv"
		labelled = "/home/felix/new_datasets/ebooks1/csv_files/labeled_data.csv"

		ebooks_df = self.select_columns(pd.read_csv(ebooks, header=0, dtype=object, na_filter=False))
		itunes_df = self.select_columns(pd.read_csv(itunes, header=0, dtype=object, na_filter=False))
		l = pd.read_csv(labelled, header=5, dtype=object, na_filter=False)

		e_clean = ebooks_df.copy()

		print itunes_df.head(1)
		print l.head(5)

		#left = itunes

		is_duplicate = l.values[:,-1]

		print is_duplicate

		print len(is_duplicate)

		for t in range(len(is_duplicate)):
			if is_duplicate[t]=='1':
				e_id = int(l.values[t,2])

				#print "Before:" + str(e_clean.values[e_id - 1,:])

				e_clean.at[e_id - 1, 'description'] = l['ltable.description'].values[t]
				e_clean.at[e_id - 1, 'price'] = l['ltable.price'].values[t]
				e_clean.at[e_id - 1, 'date'] = l['ltable.date'].values[t]
				e_clean.at[e_id - 1, 'publisher'] = l['ltable.publisher'].values[t]
				e_clean.at[e_id - 1, 'title'] = l['ltable.title'].values[t]
				e_clean.at[e_id - 1, 'author'] = l['ltable.author'].values[t]
				e_clean.at[e_id - 1, 'length'] = l['ltable.length'].values[t]

				#print "After:" + str(e_clean.values[e_id - 1, :])

		dirty_pd = ebooks_df.append(itunes_df)
		clean_pd = e_clean.append(itunes_df)

		#dirty_pd = ebooks_df
		#clean_pd = e_clean

		super(EBooks1, self).__init__(EBooks1.name, dirty_pd, clean_pd)




	def validate(self):
		print "validate"

if __name__ == '__main__':
	data = EBooks1()

	print np.sum(data.matrix_is_error, axis=0) / float(data.shape[0])

	print data.shape

	import csv
	#data.clean_pd.to_csv('/tmp/ebooks1_clean.csv', index=False, quoting=csv.QUOTE_ALL)
	#data.dirty_pd.to_csv('/tmp/ebooks1_dirty.csv', index=False, quoting=csv.QUOTE_ALL)