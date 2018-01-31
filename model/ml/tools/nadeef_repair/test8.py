from ml.datasets.blackOak import BlackOakDataSet
from ml.tools.nadeef_repair.NadeefMe import NadeefMe

tool = NadeefMe(BlackOakDataSet(), "/home/felix/SequentialPatternErrorDetection/nadeef_repair/blackoak_audit/blackoak_nadeef_new.csv")

print "Fscore: " + str(tool.calculate_total_fscore())
print "Precision: " + str(tool.calculate_total_precision())
print "Recall: " + str(tool.calculate_total_recall())