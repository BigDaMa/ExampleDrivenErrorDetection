import numpy as np
import pandas as pd

from ml.configuration.Config import Config
from ml.datasets.DataSet import DataSet


class HospitalNew(DataSet):
    name = "HospitalHoloClean"

    def __init__(self):
        path_to_dirty = "/home/felix/Software/HoloClean/tutorials/data" + "/hospital.csv"
        path_to_clean = "/home/felix/Software/HoloClean/tutorials/data" + "/hospital_clean.csv"

        dirty_wrong_format = pd.read_csv(path_to_dirty, header=0, dtype=object, na_filter=False)
        clean_wrong_format = pd.read_csv(path_to_clean, header=0, dtype=object, na_filter=False)

        clean_pd = self.to_matrix(clean_wrong_format, dirty_wrong_format.columns)

        print dirty_wrong_format.head(10)
        print dirty_wrong_format.shape

        print clean_pd.head(10)
        print clean_pd.shape



        #dirty_pd.to_csv('hospital.csv', index=False)
        #clean_pd.to_csv('hospital_clean.csv', index=False)

        super(HospitalNew, self).__init__(HospitalNew.name, dirty_wrong_format, clean_pd)


    def to_matrix(self, df, columns):
        mapColumns = {}
        for i in range(len(columns)):
            mapColumns[columns[i]] = i

        #print "shape: " + str(df.shape[0])
        #print "column: " + str(len(columns))

        pd_matrix = df.values
        matrix = np.empty([df.shape[0] / len(columns), len(columns)], dtype=object)

        for i in range(len(pd_matrix)):
            name = str(pd_matrix[i][1])
            if name in mapColumns:
                row = int(pd_matrix[i][0]) - 1
                column = mapColumns[name]
                matrix[row][column] = pd_matrix[i][2]

        newdf = pd.DataFrame(data=matrix, columns=columns)
        return newdf


    def validate(self):
        print "validate"

if __name__ == '__main__':
    data = HospitalNew()

    print np.sum(data.matrix_is_error, axis=0) / float(data.shape[0])

    print np.sum(data.matrix_is_error, axis=0)

    print np.sum(data.matrix_is_error)

    print data.shape

    np.save("/tmp/save_detected", data.matrix_is_error)