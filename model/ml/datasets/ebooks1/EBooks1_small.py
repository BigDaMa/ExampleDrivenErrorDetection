import numpy as np
import pandas as pd

from ml.configuration.Config import Config
from ml.datasets.DataSet import DataSet


class EBooks1(DataSet):
    name = "ebooks1"

    def select_columns(self, data):
        return data[['record_id', 'title', 'author', 'date', 'publisher', 'price', 'length', 'description']]

    def __init__(self):
        ebooks = "/home/felix/datasets/duplicate_data/ebooks1/csv_files/ebooks.csv"
        itunes = "/home/felix/datasets/duplicate_data/ebooks1/csv_files/itunes.csv"
        labelled = "/home/felix/datasets/duplicate_data/ebooks1/csv_files/labeled_data.csv"

        ebooks_df = self.select_columns(pd.read_csv(ebooks, header=0, dtype=object, na_filter=False))
        itunes_df = self.select_columns(pd.read_csv(itunes, header=0, dtype=object, na_filter=False))


        print str(len(ebooks_df)) + " vs " + str(len(itunes_df))

        l = pd.read_csv(labelled, header=5, dtype=object, na_filter=False)


        # left = itunes

        is_duplicate = l.values[:, -1]

        print is_duplicate

        print len(is_duplicate)

        dirty_list = []
        clean_list = []

        for t in range(len(is_duplicate)):
            if is_duplicate[t] == '1':
                e_id = int(l.values[t, 2])
                itunes_id = int(l.values[t, 1])

                print "match:" + str(l.values[t,:])

                row_e= np.where(ebooks_df["record_id"] == str(e_id))[0][0]
                row_itunes = np.where(itunes_df["record_id"] == str(itunes_id))[0][0]

                print "ebooks: " + str(ebooks_df.values[row_e, :])
                print "itunes: " + str(itunes_df.values[row_itunes, :])
                print "---"

                # print "Before:" + str(e_clean.values[e_id - 1,:])

                dirty_list.append(row_e)
                clean_list.append(row_itunes)


        print dirty_list

        dirty_pd = pd.DataFrame(data=ebooks_df.values[dirty_list, 1:ebooks_df.shape[1]], columns=itunes_df.columns[1:itunes_df.shape[1]])
        clean_pd = pd.DataFrame(data=itunes_df.values[clean_list, 1:itunes_df.shape[1]], columns=itunes_df.columns[1:itunes_df.shape[1]])

        row_number = 1
        print dirty_pd.values[row_number,:]
        print clean_pd.values[row_number,:]

        super(EBooks1, self).__init__(EBooks1.name, dirty_pd, clean_pd)

    def validate(self):
        print "validate"


if __name__ == '__main__':
    data = EBooks1()

    print np.sum(data.matrix_is_error, axis=0) / float(data.shape[0])

    print data.shape

    import csv
# data.clean_pd.to_csv('/tmp/ebooks1_clean.csv', index=False, quoting=csv.QUOTE_ALL)
# data.dirty_pd.to_csv('/tmp/ebooks1_dirty.csv', index=False, quoting=csv.QUOTE_ALL)
