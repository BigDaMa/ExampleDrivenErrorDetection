import numpy as np

from ml.datasets.blackOak import BlackOakDataSet
from ml.tools.ZiaTool import ZiaTool


class TrifactaOnBlackOak(ZiaTool):
    def __init__(self, blackOakDataSet=BlackOakDataSet()):
        path_to_tool_detected = "/home/felix/BlackOak/List_A/ToolOutputDetectedCells/Trifacta_detectedCells.csv"
        path_to_tool_correct_detected = "/home/felix/BlackOak/List_A/ToolOutputCorrectCells/Trifacta_CorrectCells.csv"
        super(TrifactaOnBlackOak, self).__init__("Trifacta", blackOakDataSet, path_to_tool_detected, path_to_tool_correct_detected)
        #self.validate()

    # validate by the results obtained from the paper "Detecting Data Errors: Where are we and what needs to be done?"
    def validate(self):
        np.testing.assert_almost_equal(0.94, self.calculate_total_fscore(), decimal=1, err_msg='', verbose=True)
        np.testing.assert_almost_equal(0.96, self.calculate_total_precision(), decimal=2, err_msg='', verbose=True)
        np.testing.assert_almost_equal(0.93, self.calculate_total_recall(), decimal=2, err_msg='', verbose=True)