import numpy as np
import pandas as pd

from ml.configuration.Config import Config
from ml.datasets.DataSet import DataSet


class Songs(DataSet):
    name = "songs"

    def __init__(self):
        msd = "/home/felix/datasets/duplicate_data/songs/msd.csv"
        labelled = "/home/felix/datasets/duplicate_data/songs/matches_msd_msd.csv"

        msd_df = pd.read_csv(msd, header=0, dtype=object, na_filter=False)
        l = pd.read_csv(labelled, header=0, dtype=object, na_filter=False)


        print msd_df.shape

        dirty_list = []
        clean_list = []

        for t in range(len(l)):
            r_id = int(l.values[t, 1])
            l_id = int(l.values[t, 0])

            print "match:" + str(l.values[t, :])

            row_l = np.where(msd_df["id"] == str(l_id))[0][0]
            row_r = np.where(msd_df["id"] == str(r_id))[0][0]

            print "amazon: " + str(msd_df.values[row_r, :])
            print "wallmart: " + str(msd_df.values[row_l, :])
            print "---"

            dirty_list.append(row_r)
            clean_list.append(row_l)

        dirty_pd = pd.DataFrame(data=msd_df.values[dirty_list, 1:msd_df.shape[1]],
                                columns=msd_df.columns[1:msd_df.shape[1]])
        clean_pd = pd.DataFrame(data=msd_df.values[clean_list, 1:msd_df.shape[1]],
                                columns=msd_df.columns[1:msd_df.shape[1]])

        row_number = 4
        print dirty_pd.values[row_number, :]
        print clean_pd.values[row_number, :]



        super(Songs, self).__init__(Songs.name, dirty_pd, clean_pd)

    def validate(self):
        print "validate"


if __name__ == '__main__':
    data = Songs()

    print np.sum(data.matrix_is_error, axis=0) / float(data.shape[0])

    print data.shape

    import csv
# data.clean_pd.to_csv('/tmp/songs_clean1.csv', index=False, quoting=csv.QUOTE_ALL)
# data.dirty_pd.to_csv('/tmp/songs_dirty1.csv', index=False, quoting=csv.QUOTE_ALL)
