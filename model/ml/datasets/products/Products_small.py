import numpy as np
import pandas as pd

from ml.configuration.Config import Config
from ml.datasets.DataSet import DataSet


class Products(DataSet):
    name = "products"

    def select_columns(self, data):
        return data[['custom_id', 'title', 'brand', 'price', 'shortdescr', 'dimensions', 'shipweight']]

    def __init__(self):
        amazon = "/home/felix/datasets/duplicate_data/products/amazon.csv"
        wallmart = "/home/felix/datasets/duplicate_data/products/walmart.csv"
        labelled = "/home/felix/datasets/duplicate_data/products/matches_walmart_amazon.csv"

        amazon_df = self.select_columns(pd.read_csv(amazon, header=0, dtype=object, na_filter=False,
                                                    names=["custom_id", "url", "asin", "brand", "modelno", "category1",
                                                           "pcategory1", "category2", "pcategory2", "title",
                                                           "listprice", "price", "prodfeatures", "techdetails",
                                                           "shortdescr", "longdescr", "dimensions", "imageurl",
                                                           "itemweight", "shipweight", "orig_prodfeatures",
                                                           "orig_techdetails"]))
        wallmart_df = self.select_columns(pd.read_csv(wallmart, header=0, dtype=object, na_filter=False,
                                                      names=["custom_id", "id", "upc", "brand", "groupname", "title",
                                                             "price", "shelfdescr", "shortdescr", "longdescr",
                                                             "imageurl", "orig_shelfdescr", "orig_shortdescr",
                                                             "orig_longdescr", "modelno", "shipweight", "dimensions"]))
        l = pd.read_csv(labelled, header=0, dtype=object, na_filter=False)


        print wallmart_df.shape
        print amazon_df.shape


        # left = wallmart

        dirty_list = []
        clean_list = []

        print wallmart_df["custom_id"]

        for t in range(len(l)):
            r_id = int(l.values[t, 1])
            l_id = int(l.values[t, 0])

            print "match:" + str(l.values[t, :])

            row_l = np.where(wallmart_df["custom_id"] == str(l_id))[0][0]
            row_r = np.where(amazon_df["custom_id"] == str(r_id))[0][0]

            print "amazon: " + str(amazon_df.values[row_r, :])
            print "wallmart: " + str(wallmart_df.values[row_l, :])
            print "---"

            dirty_list.append(row_r)
            clean_list.append(row_l)

        dirty_pd = pd.DataFrame(data=amazon_df.values[dirty_list, 1:amazon_df.shape[1]],
                                columns=wallmart_df.columns[1:wallmart_df.shape[1]])
        clean_pd = pd.DataFrame(data=wallmart_df.values[clean_list, 1:wallmart_df.shape[1]],
                                columns=wallmart_df.columns[1:wallmart_df.shape[1]])

        row_number = 4
        print dirty_pd.values[row_number, :]
        print clean_pd.values[row_number, :]

        super(Products, self).__init__(Products.name, dirty_pd, clean_pd)

    def validate(self):
        print "validate"


if __name__ == '__main__':
    data = Products()

    print np.sum(data.matrix_is_error, axis=0) / float(data.shape[0])

    print data.shape

    import csv
# data.clean_pd.to_csv('/tmp/products_clean.csv', index=False, quoting=csv.QUOTE_ALL)
# data.dirty_pd.to_csv('/tmp/products_dirty.csv', index=False, quoting=csv.QUOTE_ALL)
