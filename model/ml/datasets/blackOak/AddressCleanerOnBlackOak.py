from sklearn.metrics import recall_score

from ml.datasets.blackOak import BlackOakDataSet
from ml.tools.ZiaTool import ZiaTool


class AddressCleanerOnBlackOak(ZiaTool):
    def __init__(self, blackOakDataSet=BlackOakDataSet()):
        path_to_tool_detected = "/home/felix/BlackOak/List_A/ToolOutputDetectedCells/AddressCleaner_detectedCells.csv"
        path_to_tool_correct_detected = "/home/felix/BlackOak/List_A/ToolOutputCorrectCells/AddressCleaner_CorrectCells.csv"
        super(AddressCleanerOnBlackOak, self).__init__("AddressCleaner", blackOakDataSet,
                                                       path_to_tool_detected,
                                                       path_to_tool_correct_detected)
        #self.validate()

    # validate by the results obtained from the paper "Detecting Data Errors: Where are we and what needs to be done?"
    # does not hold
    def validate(self):
        assert self.calculate_total_precision() < 1.0

        recall = recall_score(self.dataset.matrix_is_error[0:1000].flatten(), self.matrix_detected[0:1000].flatten())

        # 0.499 != 0.61
        #np.testing.assert_almost_equal(0.61, recall, decimal=2, err_msg='', verbose=True)

    def validate_true_positives(self):
        print True