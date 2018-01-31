from ml.datasets.blackOak import BlackOakDataSet
from ml.tools.ZiaTool import ZiaTool


class DBoostGMMOnBlackOak(ZiaTool):
    def __init__(self, blackOakDataSet=BlackOakDataSet()):
        path_to_tool_detected = "/home/felix/BlackOak/List_A/ToolOutputDetectedCells/mixture.csv"
        path_to_tool_correct_detected = "/home/felix/BlackOak/List_A/ToolOutputCorrectCells/mixture_CorrectCells.csv"
        super(DBoostGMMOnBlackOak, self).__init__("dBoost_GMM", blackOakDataSet,
                                                  path_to_tool_detected,
                                                  path_to_tool_correct_detected)
        #self.validate()

    # validate by the results obtained from the paper "Detecting Data Errors: Where are we and what needs to be done?"
    # does not hold
    '''
    F-Score: 0.740158339232 != 0.38
    Precision: 0.690716892013 != 0.37
    Recall: 0.797223520442 != 0.38
    '''
    def validate(self):
        print "test"
        #np.testing.assert_almost_equal(0.38, self.calculate_total_fscore(), decimal=2, err_msg='', verbose=True)
        #np.testing.assert_almost_equal(0.38, self.calculate_total_precision(), decimal=2, err_msg='', verbose=True)
        #np.testing.assert_almost_equal(0.37, self.calculate_total_recall(), decimal=2, err_msg='', verbose=True)