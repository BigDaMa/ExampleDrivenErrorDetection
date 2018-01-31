import pandas as pd
from ml.blackOak.AddressCleanerOnBlackOak import AddressCleanerOnBlackOak
from ml.blackOak.BlackOakDataSet import BlackOakDataSet
from ml.blackOak.DBoostGMMOnBlackOak import DBoostGMMOnBlackOak
from ml.blackOak.DBoostHistogramOnBlackOak import DBoostHistogramOnBlackOak
from ml.blackOak.DCCleanOnBlackOak import DCCleanOnBlackOak
from ml.blackOak.KataraOnBlackOak import KataraOnBlackOak
from ml.blackOak.OpenRefineOnBlackOak import OpenRefineOnBlackOak
from ml.blackOak.TamrOnBlackOak import TamrOnBlackOak
from ml.blackOak.TrifactaOnBlackOak import TrifactaOnBlackOak

from ml.datasets.blackOak.DBoostGaussianOnBlackOak import DBoostGaussianOnBlackOak


class BlackOakVisualizer(object):
    def __init__(self):

        self.dataSet = BlackOakDataSet()

        self.tools = [DCCleanOnBlackOak(self.dataSet),
                 TrifactaOnBlackOak(self.dataSet),
                 OpenRefineOnBlackOak(self.dataSet),
                 DBoostGaussianOnBlackOak(self.dataSet),
                 DBoostHistogramOnBlackOak(self.dataSet),
                 DBoostGMMOnBlackOak(self.dataSet),
                 KataraOnBlackOak(self.dataSet),
                 TamrOnBlackOak(self.dataSet),
                 AddressCleanerOnBlackOak(self.dataSet)
                 ]

    def validate_tools(self):
        for tool in self.tools:
            tool.validate()
            tool.validate_true_positives()
        print "BlackOak is validated"

    def print_total_results(self, precision = 2):
        data = []
        for tool in self.tools:
            row = []
            row.append(tool.name)
            row.append(round(tool.calculate_total_precision(), precision))
            row.append(round(tool.calculate_total_recall(), precision))
            row.append(round(tool.calculate_total_fscore(), precision))
            data.append(row)

        columns = ['Tool', 'P', 'R', 'F']
        df = pd.DataFrame(data, columns=columns)

        pd.set_option('display.max_rows', len(df))
        print(df.to_string(index=False))
        pd.reset_option('display.max_rows')

    def print_results_by_all_columns(self):
        for c in range(self.dataSet.shape[1]):
            self.print_results_by_column(c)

    def print_results_by_column(self, column_id, precision = 2):
        data = []
        for tool in self.tools:
            row = []
            row.append(tool.name)
            row.append(round(tool.calculate_precision_by_column(column_id), precision))
            row.append(round(tool.calculate_recall_by_column(column_id), precision))
            row.append(round(tool.calculate_fscore_by_column(column_id), precision))
            data.append(row)

        columns = ['Tool', 'P', 'R', 'F']
        df = pd.DataFrame(data, columns=columns)

        print "column: " + str(column_id) + " - " + str(self.dataSet.dirty_pd.columns[column_id])

        pd.set_option('display.max_rows', len(df))
        print(df.to_string(index=False))
        pd.reset_option('display.max_rows')

