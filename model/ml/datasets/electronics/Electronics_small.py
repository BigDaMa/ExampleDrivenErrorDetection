import numpy as np
import pandas as pd

from ml.configuration.Config import Config
from ml.datasets.DataSet import DataSet


class Electronics(DataSet):
    name = "electronics"

    def select_columns(self, data):
        return data[["ID", "Brand", "Name", "Price", "Features"]]

    def __init__(self):
        amazon = "/home/felix/datasets/duplicate_data/electronics/csv_files/amazon.csv"
        bestbuy = "/home/felix/datasets/duplicate_data/electronics/csv_files/best_buy.csv"
        labelled = "/home/felix/datasets/duplicate_data/electronics/csv_files/labeled_data.csv"

        amazon_df = self.select_columns(pd.read_csv(amazon, header=0, dtype=object, na_filter=False,
                                                    names=["ID", "Brand", "Name", "Amazon_Price", "Price", "Features"]))
        bestbuy_df = self.select_columns(pd.read_csv(bestbuy, header=0, dtype=object, na_filter=False,
                                                     names=["ID", "Brand", "Name", "Price", "Description", "Features"]))
        l = pd.read_csv(labelled, header=5, dtype=object, na_filter=False)

        print amazon_df.shape
        print bestbuy_df.shape

        # left = amazon


        is_duplicate = l.values[:, 3]

        dirty_list = []
        clean_list = []

        for t in range(len(l)):
            if is_duplicate[t] == '1':
                r_id = int(l.values[t, 2])
                l_id = int(l.values[t, 1])

                print "match:" + str(l.values[t, :])

                row_e = np.where(amazon_df["ID"] == str(l_id))[0][0]
                row_itunes = np.where(bestbuy_df["ID"] == str(r_id))[0][0]

                print "amazon: " + str(amazon_df.values[row_e, :])
                print "bestbuy: " + str(bestbuy_df.values[row_itunes, :])
                print "---"

                dirty_list.append(row_e)
                clean_list.append(row_itunes)

        dirty_pd = pd.DataFrame(data=amazon_df.values[dirty_list, 1:amazon_df.shape[1]],
                                columns=bestbuy_df.columns[1:bestbuy_df.shape[1]])
        clean_pd = pd.DataFrame(data=bestbuy_df.values[clean_list, 1:bestbuy_df.shape[1]],
                                columns=bestbuy_df.columns[1:bestbuy_df.shape[1]])

        row_number = 4
        print dirty_pd.values[row_number, :]
        print clean_pd.values[row_number, :]

        super(Electronics, self).__init__(Electronics.name, dirty_pd, clean_pd)

    def validate(self):
        print "validate"


if __name__ == '__main__':
    data = Electronics()

    print np.sum(data.matrix_is_error, axis=0) / float(data.shape[0])

    print data.shape

    import csv
# data.clean_pd.to_csv('/tmp/electronics_clean.csv', index=False, quoting=csv.QUOTE_ALL)
# data.dirty_pd.to_csv('/tmp/electronics_dirty.csv', index=False, quoting=csv.QUOTE_ALL)
