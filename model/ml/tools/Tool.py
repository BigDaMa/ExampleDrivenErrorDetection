import pandas as pd
import numpy as np
import abc

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

class Tool(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, tool_name, dataset, matrix_detected):
        self.name = tool_name
        self.dataset = dataset

        self.matrix_detected = matrix_detected

        #cache here

    @abc.abstractmethod
    def validate(self):
        """validate whether we loaded the data set correctly"""
        return


    def write_detected_matrix(self, output_path='/tmp/matrix_detected.npy'):
        np.save(output_path, self.matrix_detected)

    def validate_positives(self):
        print "hello"

    def calculate_total_fscore(self):
        return f1_score(self.dataset.matrix_is_error.flatten(), self.matrix_detected.flatten())

    def calculate_total_precision(self):
        return precision_score(self.dataset.matrix_is_error.flatten(), self.matrix_detected.flatten())

    def calculate_total_recall(self):
        return recall_score(self.dataset.matrix_is_error.flatten(), self.matrix_detected.flatten())

    def calculate_fscore_by_column(self, column_id):
        return f1_score(self.dataset.matrix_is_error[:,column_id].flatten(), self.matrix_detected[:,column_id].flatten())

    def calculate_precision_by_column(self, column_id):
        return precision_score(self.dataset.matrix_is_error[:,column_id].flatten(), self.matrix_detected[:,column_id].flatten())

    def calculate_recall_by_column(self, column_id):
        return recall_score(self.dataset.matrix_is_error[:,column_id].flatten(), self.matrix_detected[:,column_id].flatten())

    def calculate_total_confusion_matrix(self):
        A = self.dataset.matrix_is_error.flatten()
        B = self.matrix_detected.flatten()

        print A
        print B


        matrix = confusion_matrix(A,B)
        if len(matrix) == 1:
            return matrix[0], 0, 0, 0

        tn, fp, fn, tp = matrix.ravel()
        return tn, fp, fn, tp

