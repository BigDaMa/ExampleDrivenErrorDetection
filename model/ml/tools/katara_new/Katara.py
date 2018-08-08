import numpy as np
import pandas as pd

from ml.tools.Tool import Tool


class Katara(Tool):

    def __init__(self, path_to_tool_result, data):
        outliers = pd.read_csv(path_to_tool_result, header=None, sep=',', dtype=int)

        print "detected: " + str(outliers.shape)

        matrix_detected = np.zeros(data.shape, dtype=bool)

        for i in range(outliers.shape[0]):
            matrix_detected[outliers.values[i,0]-1, outliers.values[i,1]] = True

        super(Katara, self).__init__("Katara_me", data, matrix_detected)


    def validate(self):
        print "test"