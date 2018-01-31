import numpy as np
import pandas as pd

from ml.tools.Tool import Tool


class ZiaTool(Tool):

    def __init__(self, tool_name, dataset, path_to_tool_detected, path_to_tool_correct_detected):
        self.path_to_tool_detected = path_to_tool_detected
        self.path_to_tool_correct_detected = path_to_tool_correct_detected

        matrix_detected = self.load_matrix_from_csv(self.path_to_tool_detected, dataset)

        super(ZiaTool, self).__init__(tool_name, dataset, matrix_detected)

        #cache here

    def load_matrix_from_csv(self,path, dataset):
        csv_list = pd.read_csv(path, header=None)
        matrix = np.zeros((dataset.shape[0], dataset.shape[1]), dtype=bool)

        for index, row in csv_list.iterrows():
            row_id = row[csv_list.columns[0]] - 1
            column_id = row[csv_list.columns[1]] - 1
            matrix[row_id][column_id] = True

        return matrix



    def validate_true_positives(self):
        matrix_correct_detected = self.load_matrix_from_csv(self.path_to_tool_correct_detected)
        my_correct_detected = np.logical_and(self.dataset.matrix_is_error, self.matrix_detected)

        index = np.where(my_correct_detected != matrix_correct_detected)
        for t in range(len(index[0])):
            print str(self.dataset.dirty_pd.values[index[0][t],index[1][t]]) + " <> " + str(self.dataset.clean_pd.values[index[0][t],index[1][t]])

        is_valid = np.all(my_correct_detected == matrix_correct_detected)
        assert is_valid, "True positives are calculated differently!"

