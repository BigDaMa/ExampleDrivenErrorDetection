import pandas as pd
import numpy as np

from ml.datasets.DataSet import DataSet


class Restaurant(DataSet):
    name = "Restaurant"
    def __init__(self):
        path_to_mohammad_git_repo = "/home/felix/SequentialPatternErrorDetection/luna/restaurant/"

        path_dirty = path_to_mohammad_git_repo + "dirty/restaurants_2009_3_12.txt"
        path_clean = path_to_mohammad_git_repo + "restaurants_golden.txt"

        dirty_pd = pd.read_csv(path_dirty, sep='\t', header=None, dtype=object, na_filter=False, names=['Source','Restaurant','Address'])
        clean_restaurants = pd.read_csv(path_clean, sep='\t', header=None, dtype=object, na_filter=False, names=['Restaurant', 'Open'])


        restaurant_map = {}
        for i_restaurant in range(clean_restaurants.shape[0]):
            restaurant_map[clean_restaurants.values[i_restaurant,0]] = clean_restaurants.values[i_restaurant,1]

        golden_restaurants = restaurant_map.keys()

        dirty_pd['Restaurant'] = dirty_pd['Restaurant'].str.lower()
        dirty_pd = dirty_pd[dirty_pd['Restaurant'].isin(golden_restaurants)]

        rest = dirty_pd['Restaurant'].copy()
        rest.replace(restaurant_map, inplace=True)

        dirty_pd['Open'] = rest
        dirty_pd = dirty_pd.sort_values(['Restaurant', 'Source'], ascending=[1, 1])

        clean_pd = dirty_pd.copy()
        dirty_pd['Open'].replace({'N':'Y'}, inplace=True)


        super(Restaurant, self).__init__(Restaurant.name, dirty_pd, clean_pd)

    def validate(self):
        print "validate"

if __name__ == '__main__':

    data = Restaurant()

    print np.sum(data.matrix_is_error)