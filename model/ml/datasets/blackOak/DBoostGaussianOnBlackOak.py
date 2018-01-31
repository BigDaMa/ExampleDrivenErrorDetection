from ml.datasets.blackOak import BlackOakDataSet
from ml.tools.ZiaTool import ZiaTool


class DBoostGaussianOnBlackOak(ZiaTool):
    def __init__(self, blackOakDataSet=BlackOakDataSet()):
        path_to_tool_detected = "/home/felix/BlackOak/List_A/ToolOutputDetectedCells/gaussian.csv"
        path_to_tool_correct_detected = "/home/felix/BlackOak/List_A/ToolOutputCorrectCells/gaussian_CorrectCells.csv"
        super(DBoostGaussianOnBlackOak, self).__init__("dBoost_Gaussian", blackOakDataSet,
                                                       path_to_tool_detected,
                                                       path_to_tool_correct_detected)
        #self.validate()

    # validate by the results obtained from the paper "Detecting Data Errors: Where are we and what needs to be done?"
    # does not hold
    '''
    F-Score: 0.728281022864 != 0.81
    Precision: 0.606291160186 != 0.91
    Recall: 0.911726540001 != 0.73
    '''
    def validate(self):
        print "test"
        #np.testing.assert_almost_equal(0.38, self.calculate_total_fscore(), decimal=2, err_msg='', verbose=True)
        #np.testing.assert_almost_equal(0.38, self.calculate_total_precision(), decimal=2, err_msg='', verbose=True)
        #np.testing.assert_almost_equal(0.37, self.calculate_total_recall(), decimal=2, err_msg='', verbose=True)