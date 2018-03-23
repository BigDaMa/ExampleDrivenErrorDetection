import numpy as np
import pandas as pd

from ml.configuration.Config import Config
from ml.datasets.DataSet import DataSet


class Songs(DataSet):
	name = "songs"

	def __init__(self):
		msd = "/home/felix/new_datasets/songs/msd.csv"
		labelled = "/home/felix/new_datasets/songs/matches_msd_msd.csv"

		msd_df = pd.read_csv(msd, header=0, dtype=object, na_filter=False)
		l = pd.read_csv(labelled, header=0, dtype=object, na_filter=False)

		msd_clean = msd_df.copy()

		#left = wallmart

		left_ids = l.values[:, 0]
		right_ids = l.values[:, 1]

		import networkx as nx
		G = nx.Graph()
		for t in range(len(left_ids)):
			G.add_edge(int(left_ids[t]), int(right_ids[t]))
			G.add_edge(int(right_ids[t]), int(left_ids[t]))

		all_components = 0
		cc = nx.connected_components(G)

		mapNodeToRoot = {}

		for g in cc:
			#find root
			g_list = list(g)
			root = g_list[0]
			if len(g) > 1:
				print(g)
				for l_counter in range(len(g_list)):
					if not "\\" in str(msd_clean.values[g_list[l_counter] - 1, :]):
						root = g_list[l_counter]
						break
				all_components += 1
			#register all
			for node in g:
				mapNodeToRoot[node] = root

		print "all " + str(all_components)

		counter = 0
		for t in range(len(left_ids)):

			#print "Before:" + str(msd_clean.values[int(right_ids[t]) - 1, :])

			for col in msd_clean.columns:
				#msd_clean.at[int(right_ids[t]) - 1, col] = msd_df[col].values[int(left_ids[t]) - 1]
				msd_clean.at[int(right_ids[t]) - 1, col] = msd_df[col].values[mapNodeToRoot[int(right_ids[t])] - 1]
				msd_clean.at[int(left_ids[t]) - 1, col] = msd_df[col].values[mapNodeToRoot[int(left_ids[t])] - 1]

			#print "After:" + str(msd_clean.values[int(right_ids[t]) - 1, :])
			counter+=1

			#if counter > 100:
			#	break

		dirty_pd = msd_df
		clean_pd = msd_clean

		print "done first"

		super(Songs, self).__init__(Songs.name, dirty_pd, clean_pd)




	def validate(self):
		print "validate"

if __name__ == '__main__':
	data = Songs()

	print np.sum(data.matrix_is_error, axis=0) / float(data.shape[0])

	print data.shape

	import csv
	#data.clean_pd.to_csv('/tmp/songs_clean1.csv', index=False, quoting=csv.QUOTE_ALL)
	#data.dirty_pd.to_csv('/tmp/songs_dirty1.csv', index=False, quoting=csv.QUOTE_ALL)