import numpy as np
import pandas as pd

from ml.tools.Tool import Tool


class OpenRefine(Tool):

    def __init__(self, path_to_tool_result, data):
        outliers = pd.read_csv(path_to_tool_result, header=0, sep='\t', dtype=object)
        outliers = self.fillna_df(outliers)

        print outliers.shape
        print data.dirty_pd.shape
        assert np.array_equal(outliers.shape, data.dirty_pd.shape), "The outliers and the dirty data shape is not equal!"

        matrix_detected = outliers.values != data.dirty_pd.values  # all detected errors

        super(OpenRefine, self).__init__("OpenRefine_me", data, matrix_detected)


    def validate(self):
        print "test"

    def fillna_df(self, df):
        for i in range(df.shape[1]):
            if df[df.columns[i]].dtypes.name == "object":
                df[df.columns[i]] = df[df.columns[i]].fillna('')
            else:
                raise Exception('not implemented')
            #todo if numeric
        return df