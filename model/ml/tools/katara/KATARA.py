import numpy as np
import pandas as pd

from ml.tools.Tool import Tool

from cleaning_api import install_tools
from cleaning_api import run_data_cleaning_job

from ml.configuration.Config import Config



class KATARA(Tool):

	def __init__(self, data):
		new_columns = []
		for col_i in range(len(data.dirty_pd.columns)):
			new_columns.append(data.dirty_pd.columns[col_i].replace(" ", "_"))

		data.dirty_pd.columns = new_columns

		print data.dirty_pd.columns

		data.dirty_pd.to_csv('/tmp/data.csv', index=False)




		'''
		data.dirty_pd.to_csv('/tmp/data.csv',
					 index=False,
					 quoting=csv.QUOTE_ALL,
					 escapechar='\\',
					 quotechar="'",
					 na_rep="")
		'''
		#/home/felix/abstractionlayer/datasets
		#/tmp/data.csv
		run_input = {
			"dataset": {
				"type": "csv",
				"param": ["/home/felix/abstractionlayer/datasets/hosp_holoclean.csv"]
			},
			"tool": {
				"name": "katara",
				"param": [Config.get("abstractionlayer.tools") + "/KATARA/dominSpecific"]
			}
		}

		matrix_detected = np.zeros(data.shape)

		results_list = run_data_cleaning_job(run_input)
		for x in results_list:
			print x
			matrix_detected[x[0]-1, x[1]] = True



		super(KATARA, self).__init__("KATARA_me", data, matrix_detected)


	def validate(self):
		print "test"

if __name__ == '__main__':
    KATARA()