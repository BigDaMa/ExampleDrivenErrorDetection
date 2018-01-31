import numpy as np
from ml.datasets.DataSetBasic import DataSetBasic
import numpy as np

from ml.datasets.DataSetBasic import DataSetBasic


class DataSet(DataSetBasic):

    def __init__(self, name, dirty_pd, clean_pd):
        self.clean_pd = self.fillna_df(clean_pd)
        dirty_pd = self.fillna_df(dirty_pd)

        assert np.array_equal(dirty_pd.shape, clean_pd.shape), "The clean and the dirty data shape is not equal!"

        matrix_is_error = dirty_pd.values != self.clean_pd.values  # all real errors

        super(DataSet, self).__init__(name, dirty_pd, matrix_is_error)