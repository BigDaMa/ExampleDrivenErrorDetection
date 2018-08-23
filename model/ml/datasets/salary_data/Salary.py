import numpy as np
import pandas as pd

from ml.datasets.DataSet import DataSet
from ml.configuration.Config import Config

class Salary(DataSet):
    name = "Salary"

    def __init__(self):
        path_to_dirty = Config.get("datapool.folder") + "/SALARIES/salaries_full/dirty/dirty_salaries_full_with_id.csv"
        path_to_clean = Config.get("datapool.folder") + "/SALARIES/salaries_full/salaries_with_id.csv"

        dirty_pd = pd.read_csv(path_to_dirty, header=0, dtype=object, error_bad_lines=False, na_filter=False)
        clean_pd = pd.read_csv(path_to_clean, header=0, dtype=object, error_bad_lines=False, na_filter=False)

        dirty_pd = dirty_pd.sort_values(['oid', 'id'], ascending=[1,1])
        clean_pd = clean_pd.sort_values(['oid', 'id'], ascending=[1,1])

        dirty_pd = dirty_pd[dirty_pd['oid'].isin(clean_pd['oid'].unique())]
        clean_pd = clean_pd[clean_pd['oid'].isin(dirty_pd['oid'].unique())]

        dirty_pd.drop('notes', axis=1, inplace=True)
        clean_pd.drop('notes', axis=1, inplace=True)

        dirty_pd = dirty_pd.reset_index(drop=True)
        clean_pd = clean_pd.reset_index(drop=True)

        assert np.all(dirty_pd['oid'] == clean_pd['oid'])
        assert np.all(dirty_pd['id'] == clean_pd['id'])
        assert np.all(dirty_pd['employeename'] == clean_pd['employeename'])
        assert np.all(dirty_pd['jobtitle'] == clean_pd['jobtitle'])
        assert np.all(dirty_pd['overtimepay'] == clean_pd['overtimepay'])
        assert np.all(dirty_pd['otherpay'] == clean_pd['otherpay'])
        assert np.all(dirty_pd['benefits'] == clean_pd['benefits'])
        assert np.all(dirty_pd['totalpaybenefits'] == clean_pd['totalpaybenefits'])
        assert np.all(dirty_pd['year'] == clean_pd['year'])
        assert np.all(dirty_pd['agency'] == clean_pd['agency'])
        assert np.all(dirty_pd['status'] == clean_pd['status'])


        super(Salary, self).__init__("Salary", dirty_pd, clean_pd)


    def validate(self):
        print "validate"

if __name__ == '__main__':
    data = Salary()

    print data.clean_pd.columns

    print data.clean_pd.values[200,:]
    print data.dirty_pd.values[200,:]