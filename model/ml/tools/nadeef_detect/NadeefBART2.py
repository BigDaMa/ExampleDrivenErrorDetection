from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.tools.katara_new.Katara import Katara

from ml.datasets.BartDataset.BartDataSet import BartDataset
data = BartDataset(BlackOakDataSetUppercase(), "CityFD_10percent")

#data.dirty_pd.to_csv('/tmp/data.csv', index=False)

tool = Katara("/home/felix/abstractionlayer/log.txt", data)

print data.shape

tool.matrix_detected[:,7]=False

print "Fscore: " + str(tool.calculate_total_fscore())
print "Precision: " + str(tool.calculate_total_precision())
print "Recall: " + str(tool.calculate_total_recall())

for c in range(data.shape[1]):
    print tool.calculate_fscore_by_column(c)