import pandas as pd
import numpy as np

class DataSetBasic(object):

    def __init__(self, name, dirty_pd, matrix_is_error):
        self.name = name
        self.dirty_pd = self.fillna_df(dirty_pd)

        self.shape = self.dirty_pd.shape

        assert np.all(matrix_is_error.shape == dirty_pd.shape)

        self.matrix_is_error = matrix_is_error  # all real errors

    def fillna_df(self, df):
        for i in range(df.shape[1]):
            if df[df.columns[i]].dtypes.name == "object":
                df[df.columns[i]] = df[df.columns[i]].fillna('')
            else:
                raise Exception('not implemented')
            #todo if numeric
        return df

    def get_number_dirty_columns(self):
        errors_per_column = np.sum(self.matrix_is_error, axis=0)
        return len(np.where(errors_per_column > 0)[0])

    def is_column_applicable(self, column_id):
        return len(np.where(self.matrix_is_error[:, column_id] == True)[0]) >= 2 and len(
                    np.where(self.matrix_is_error[:, column_id] == False)[0]) >= 2

    def get_applicable_columns(self):
        my_list = []
        for col_i in range(self.shape[1]):
            if self.is_column_applicable(col_i):
                my_list.append(col_i)
        return my_list

    def get_number_applicable_columns(self):
        return len(self.get_applicable_columns())

