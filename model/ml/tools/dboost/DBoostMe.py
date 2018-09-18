import numpy as np
import pandas as pd

from ml.tools.Tool import Tool
import os

class DBoostMe(Tool):
    def __init__(self, dataSet, path_to_tool_result="/home/felix/SequentialPatternErrorDetection/dboost/outputDBoost-blackoak-gaus-new16875.csv"):
        #path_to_tool_result = "/home/felix/SequentialPatternErrorDetection/dboost/repaired/outputDBoost-blackoak--histogram 0.9 0.01 --discretestats 8 2 -d string_case.csv"
        #command = "./dboost-stdin.py -F ','  --histogram 0.9 0.01 --discretestats 8 2 -d string_case /home/felix/BlackOak/List_A/inputDB.csv > /home/felix/SequentialPatternErrorDetection/dboost/outputDBoost-blackoak.csv"

        matrix_detected = np.zeros(dataSet.shape, dtype=bool)

        try:
            outliers = pd.read_csv(path_to_tool_result, sep='|', na_filter=False, error_bad_lines=False, header=None)
        except pd.io.common.EmptyDataError:
            super(DBoostMe, self).__init__("DBoost_me", dataSet, matrix_detected)
            return
        outliers = self.fillna_df(outliers)

        #print outliers

        matrix_outliers = outliers.values

        os.remove(path_to_tool_result)


        for i in range(len(matrix_outliers)):
            if int(matrix_outliers[i][0]) > 0:
                row_id = int(matrix_outliers[i][0])-1
                attribute_id = int(matrix_outliers[i][1])
                old_value = matrix_outliers[i][2]

                #print str(old_value) + " vs " + str(dataSet.dirty_pd.values[row_id][attribute_id])
                #assert old_value == dataSet.dirty_pd.values[row_id][attribute_id], '#' + str(old_value) + "# vs our: #" + str(dataSet.dirty_pd.values[row_id][attribute_id] + "#")
                matrix_detected[row_id][attribute_id] = True

        super(DBoostMe, self).__init__("DBoost_me", dataSet, matrix_detected)

    def validate(self):
        print "test"

    def fillna_df(self, df):
        df[df.columns[2]] = df[df.columns[2]].fillna('')
        #todo if numeric
        return df
