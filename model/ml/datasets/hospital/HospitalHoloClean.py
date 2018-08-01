import numpy as np
import pandas as pd

from ml.configuration.Config import Config
from ml.datasets.DataSet import DataSet


class HospitalHoloClean(DataSet):
    name = "HospitalHoloClean"

    def __init__(self):
        path_to_dirty = Config.get("datapool.folder") + "/HOSP_HoloClean/dirty/hospital_input.csv"
        path_to_clean = Config.get("datapool.folder") + "/HOSP_HoloClean/ground-truth/hospital_clean.csv"

        dirty_wrong_format = pd.read_csv(path_to_dirty, header=0, dtype=object)
        clean_wrong_format = pd.read_csv(path_to_clean, header=0, dtype=object)

        dirty_pd = self.to_matrix(dirty_wrong_format)
        clean_pd = self.to_matrix(clean_wrong_format)

        # remove empty columns
        dirty_pd = dirty_pd.drop(['address2', 'address3'], 1)
        clean_pd = clean_pd.drop(['address2', 'address3'], 1)


        #dirty_pd.to_csv('hospital.csv', index=False)
        #clean_pd.to_csv('hospital_clean.csv', index=False)

        super(HospitalHoloClean, self).__init__(HospitalHoloClean.name, dirty_pd, clean_pd)


    def to_matrix(self, df):
        columns = ['provider number', 'hospital name','address1', 'address2', 'address3', 'city','state','zip code',
                  'county name', 'phone number', 'hospital type', 'hospital owner', 'emergency service', 'condition',
                  'measure code','measure name', 'score', 'sample', 'stateavg']

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
