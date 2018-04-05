import numpy as np
import pandas as pd

from ml.configuration.Config import Config
from ml.datasets.DataSet import DataSet


class MyFD(DataSet):
    name = "MyFD"

    def __init__(self, dataset, number_errors, column_name):

        rng = np.random.RandomState(42)

        clean_pd = dataset.clean_pd
        dirty_pd = clean_pd.copy()

        if number_errors < 1.0:
            number_errors = int(clean_pd.shape[0] * number_errors)


        cities_set = set(clean_pd[column_name].unique())

        ids = set()

        error_count=0
        while error_count < number_errors:
            cities = list(cities_set)
            city = rng.randint(len(cities))
            row = rng.randint(clean_pd.shape[0])

            #print row

            if row in ids:
                continue
            ids.add(row)

            if dirty_pd[column_name].values[row] != cities[city]:
                #dirty_pd.at['city', row] = cities[city]
                print str(dirty_pd[column_name][row]) + "->" + str(cities[city])
                if dirty_pd[column_name][row] in cities_set:
                    cities_set.remove(dirty_pd[column_name][row])
                dirty_pd[column_name][row] = cities[city]

                error_count += 1


        #print dirty_pd.shape
        #print clean_pd.shape

        #dirty_pd.to_csv('hospital.csv', index=False)
        #clean_pd.to_csv('hospital_clean.csv', index=False)

        super(MyFD, self).__init__(MyFD.name, dirty_pd, clean_pd)


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

if __name__ == '__main__':
    from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
    data = MyFD(HospitalHoloClean(), 0.01, "city")
    #from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
    #data = MyFD(BlackOakDataSetUppercase(), 0.01, "City")

    print np.sum(data.matrix_is_error, axis=0) / float(data.shape[0])
    print np.sum(data.matrix_is_error, axis=0)

    print data.shape
