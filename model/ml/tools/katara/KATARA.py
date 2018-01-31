import numpy as np
import pandas as pd

from ml.tools.Tool import Tool

from cleaning_api import install_tools
from cleaning_api import run_data_cleaning_job



class KATARA(Tool):

    def __init__(self, path_to_tool_result, data):
		install_tools()

		run_input = {
			"dataset": {
				"type": "csv",
				"param": ["/tmp/hospital.csv"]
			},
			"tool": {
				"name": "katara",
				"param": ["tools/KATARA/dominSpecific"]
			}
		}

		results_list = run_data_cleaning_job(run_input)
		for x in results_list:
			print x


        #matrix_detected = outliers.values != data.dirty_pd.values  # all detected errors

        #super(KATARA, self).__init__("KATARA_me", data, matrix_detected)


    def validate(self):
        print "test"

    def fillna_df(self, df):
        for i in range(df.shape[1]):
            if df[df.columns[i]].dtypes.name == "object":
                df[df.columns[i]] = df[df.columns[i]].fillna('')
            else:
                raise Exception('not implemented')
            #todo if numeric
        return df

if __name__ == '__main__':
    KATARA()