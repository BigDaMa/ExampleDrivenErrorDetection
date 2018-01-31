from ml.datasets.blackOak.BlackOakDataSet import BlackOakDataSet
from ml.tools.dboost.DBoostMe import DBoostMe

tool = DBoostMe(BlackOakDataSet(), "/tmp/test_format.csv")

print "Fscore: " + str(tool.calculate_total_fscore())
print "Precision: " + str(tool.calculate_total_precision())
print "Recall: " + str(tool.calculate_total_recall())

#data = BlackOakDataSet()

#data.dirty_pd.to_csv("blackOak_clear.csv",index=False, quotechar="\"")