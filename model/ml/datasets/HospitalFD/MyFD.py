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


        cities = list(clean_pd[column_name].unique())

        zips = list(clean_pd['zip code'].unique())


        zip_dict = {}
        for row_cur in range(dataset.shape[0]):
            zip_cur = clean_pd['zip code'][row_cur]
            if not zip_cur in zip_dict:
                zip_dict[zip_cur] = set()
            zip_dict[zip_cur].add(row_cur)


        ids = set()

        error_count = 0
        while error_count < number_errors:
            city = rng.randint(len(cities))
            row = rng.randint(clean_pd.shape[0])

            #print row

            if row in ids:
                continue
            ids.add(row)

            if len(zip_dict[dirty_pd['zip code'].values[row]]) > 1:
                print str(dirty_pd[column_name][row]) + "->" + str(cities[city])
                zip_dict[dirty_pd['zip code'].values[row]].remove(row)
                dirty_pd[column_name][row] = cities[city]

                error_count += 1

        super(MyFD, self).__init__(MyFD.name, dirty_pd, clean_pd)

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
