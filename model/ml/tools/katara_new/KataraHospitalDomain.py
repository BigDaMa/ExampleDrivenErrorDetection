from sets import Set

from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.tools.katara_new.Katara import Katara

from ml.datasets.HospitalDomainError.HospitalDomainError import HospitalDomainError
data = HospitalDomainError()

tool = Katara("/home/felix/datasets/hospital_domain/log.txt", data)

print "Fscore: " + str(tool.calculate_total_fscore())
print "Precision: " + str(tool.calculate_total_precision())
print "Recall: " + str(tool.calculate_total_recall())

for c in range(data.shape[1]):
    print tool.calculate_fscore_by_column(c)