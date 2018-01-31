import numpy as np

from ml.datasets.blackOak import BlackOakDataSet
from ml.tools.ZiaTool import ZiaTool


class DBoostHistogramOnBlackOak(ZiaTool):
    def __init__(self, blackOakDataSet=BlackOakDataSet()):
        path_to_tool_detected = "/home/felix/BlackOak/List_A/ToolOutputDetectedCells/histograms.csv"
        path_to_tool_correct_detected = "/home/felix/BlackOak/List_A/ToolOutputCorrectCells/histograms_CorrectCells.csv"
        super(DBoostHistogramOnBlackOak, self).__init__("dBoost_Histogram", blackOakDataSet,
                                                        path_to_tool_detected,
                                                        path_to_tool_correct_detected)
        #self.validate()

    # validate by the results obtained from the paper "Detecting Data Errors: Where are we and what needs to be done?"
    def validate(self):
        print "test"
        np.testing.assert_almost_equal(0.52, self.calculate_total_fscore(), decimal=1, err_msg='', verbose=True)
        np.testing.assert_almost_equal(0.52, self.calculate_total_precision(), decimal=2, err_msg='', verbose=True)
        np.testing.assert_almost_equal(0.51, self.calculate_total_recall(), decimal=2, err_msg='', verbose=True)