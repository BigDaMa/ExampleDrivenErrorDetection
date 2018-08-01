import numpy as np
import pandas as pd

from ml.configuration.Config import Config
from ml.datasets.DataSet import DataSet


class FlightHoloClean(DataSet):
    name="Flight HoloClean"

    def __init__(self):
        path_to_dirty = Config.get("datapool.folder") + "/FLIGHTS_HoloClean/dirty/flights_input.csv"
        path_to_clean = Config.get("datapool.folder") + "/FLIGHTS_HoloClean/ground-truth/flights_clean.csv"

        dirty_wrong_format = pd.read_csv(path_to_dirty, header=0, dtype=object)
        clean_wrong_format = pd.read_csv(path_to_clean, header=0, dtype=object)

        dirty_pd = self.to_matrix(dirty_wrong_format)
        clean_pd = self.to_matrix(clean_wrong_format)

        dirty_pd = dirty_pd.sort_values(['flight', 'src'], ascending=[1,1])
        clean_pd = clean_pd.sort_values(['flight', 'src'], ascending=[1,1])

        assert np.all(dirty_pd['flight'] == clean_pd['flight'])
        assert np.all(dirty_pd['src'] == clean_pd['src'])

        super(FlightHoloClean, self).__init__(FlightHoloClean.name, dirty_pd, clean_pd)


    def to_matrix(self, df):

        columns = ['src', 'flight', 'sched_dep_time', 'act_dep_time', 'sched_arr_time', 'act_arr_time']

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
        return newdf


    def validate(self):
        print "validate"
