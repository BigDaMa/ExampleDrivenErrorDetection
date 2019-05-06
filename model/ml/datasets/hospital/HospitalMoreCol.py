import numpy as np
import pandas as pd

from ml.configuration.Config import Config
from ml.datasets.DataSet import DataSet
import copy


class HospitalMoreCol(DataSet):
    name="HospitalMoreCol"

    def __init__(self):
        path_to_dirty = Config.get("datapool.folder") + "/HOSP_HoloClean/dirty/hospital_input.csv"
        path_to_clean = Config.get("datapool.folder") + "/HOSP_HoloClean/ground-truth/hospital_clean.csv"

        dirty_wrong_format = pd.read_csv(path_to_dirty, header=0, dtype=object, na_filter=False)
        clean_wrong_format = pd.read_csv(path_to_clean, header=0, dtype=object, na_filter=False)


        columns = np.unique(dirty_wrong_format['attribute'].values)
        #print(len(columns))

        dirty_pd, mapColumns = self.to_matrix(dirty_wrong_format, columns)
        clean_pd = self.correct_dirty(mapColumns, dirty_pd, clean_wrong_format)

        #print(dirty_pd.head())

        super(HospitalMoreCol, self).__init__(HospitalMoreCol.name, dirty_pd, clean_pd)


    def to_matrix(self, df, columns):

        mapColumns = {}
        for i in range(len(columns)):
            mapColumns[columns[i]] = i

        pd_matrix = df.values
        matrix = np.empty([df.shape[0] / len(columns), len(columns)], dtype=object)

        for i in range(len(pd_matrix)):
            row = int(pd_matrix[i][0]) - 1
            column = mapColumns[str(pd_matrix[i][1])]
            matrix[row][column] = pd_matrix[i][2]

        newdf = pd.DataFrame(data=matrix, columns=columns)
        return newdf, mapColumns

    def correct_dirty(self, mapColumns, dirty_pd, clean_wrong_format):
        dirty_data = copy.deepcopy(dirty_pd.values)

        pd_matrix = clean_wrong_format.values
        for i in range(len(pd_matrix)):
            row = int(pd_matrix[i][0]) - 1
            column = mapColumns[str(pd_matrix[i][1])]
            #print('ID: ' + str(pd_matrix[i][0]))

            #if dirty_data[row][column] != pd_matrix[i][2]:
            #    print(" original: #" + str(dirty_data[row][column]) + '#')
            #    print("corrected: #" + str(pd_matrix[i][2]) + '#\n\n')
            dirty_data[row][column] = pd_matrix[i][2]
        return pd.DataFrame(data=dirty_data, columns=dirty_pd.columns)



    def validate(self):
        print "validate"

if __name__ == "__main__":
    HospitalMoreCol()