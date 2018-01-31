import numpy as np
import pandas as pd

from ml.datasets.DataSet import DataSet


class Salary(DataSet):
    name = "Salary"

    def __init__(self):
        path_to_dirty = "/home/felix/data-pool/SALARIES/salaries_full/dirty/dirty_salaries_full_with_id.csv"
        path_to_clean = "/home/felix/data-pool/SALARIES/salaries_full/salaries_with_id.csv"

        dirty_pd = pd.read_csv(path_to_dirty, header=0, dtype=object, error_bad_lines=False, na_filter=False)
        clean_pd = pd.read_csv(path_to_clean, header=0, dtype=object, error_bad_lines=False, na_filter=False)

        dirty_pd = dirty_pd.sort(['oid', 'id'], ascending=[1,1])
        clean_pd = clean_pd.sort(['oid', 'id'], ascending=[1,1])

        dirty_pd = dirty_pd[dirty_pd['oid'].isin(clean_pd['oid'].unique())]
        clean_pd = clean_pd[clean_pd['oid'].isin(dirty_pd['oid'].unique())]

        dirty_pd.drop('notes', axis=1, inplace=True)
        clean_pd.drop('notes', axis=1, inplace=True)

        dirty_pd = dirty_pd.reset_index(drop=True)
        clean_pd = clean_pd.reset_index(drop=True)

        assert np.all(dirty_pd['oid'] == clean_pd['oid'])
        assert np.all(dirty_pd['id'] == clean_pd['id'])

        super(Salary, self).__init__("Salary", dirty_pd, clean_pd)


    def validate(self):
        print "validate"
