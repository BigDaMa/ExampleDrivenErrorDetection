import pickle

import numpy as np
import pandas as pd

from ml.datasets.DataSet import DataSet


class Products(DataSet):
    def __init__(self):
        pairs = pickle.load( open( "/home/felix/SequentialPatternErrorDetection/DQM_datasets/product/responses.p", "rb" ) )

        print len(pairs)

        data = pd.read_csv("/home/felix/SequentialPatternErrorDetection/DQM_datasets/product/products.csv", header=None)


        ground_truth = pd.read_csv("/home/felix/SequentialPatternErrorDetection/DQM_datasets/product/product_mapping.csv", header=None)

        mappings = {}
        for i in range(len(ground_truth)):
            mappings[ground_truth.values[i,0]] = ground_truth.values[i,1]
            mappings[ground_truth.values[i,1]] = ground_truth.values[i,0]


        data_dict = {}
        for tuple in data.values:
            data_dict[tuple[1]] = tuple

        #print data_dict

        def remove(str):
            return str.replace('"', '')

        data_pairs = []

        target = []

        for pair in pairs:
            my_list = []
            id0 = remove(pair[0][1])
            id1 = remove(pair[0][0])
            #print data_dict[id1]

            my_list.extend(data_dict[id0])
            my_list.extend(data_dict[id1])

            data_pairs.append(my_list)

            if (my_list[1] in mappings and mappings[my_list[1]] == my_list[7]) or (my_list[7] in mappings and mappings[my_list[7]] == my_list[1]):
                target.append(1)
            else:
                target.append(0)


        all_data1 = np.matrix(data_pairs)

        all_data_dirty = np.hstack((all_data1, np.zeros((all_data1.shape[0],1), dtype=int)))

        all_data_clean = np.hstack((all_data1, np.matrix(target).transpose()))


        #print np.where(target)[0]
        #print len(np.where(target)[0])

        dirty_pd = pd.DataFrame(self.clean_codec(all_data_dirty), dtype=object)
        clean_pd = pd.DataFrame(self.clean_codec(all_data_clean), dtype=object)


        super(Products, self).__init__("Flight HoloClean", dirty_pd, clean_pd)

    def validate(self):
        print "validate"

    def clean_codec(self, matrix):
        for x in range(len(matrix)):
            for y in range(matrix.shape[1]):
                matrix[x, y] = unicode(str(matrix[x,y]), errors='ignore')
        return matrix