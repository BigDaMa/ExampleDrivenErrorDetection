import pandas as pd
import numpy as np
import abc

class Hospital(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.name = "Hospital"
        self.dirty_pd = pd.read_csv("/home/felix/NADEEF/examples/hospital500k.csv", sep=',', header=0)

        self.shape = self.dirty_pd.shape

        f = open('/home/felix/SequentialPatternErrorDetection/nadeef_repair/matrix_audit/matrix_is_error', 'r')

        self.matrix_is_error = np.load(f)

        print self.matrix_is_error.shape