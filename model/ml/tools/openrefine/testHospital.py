from ml.datasets.hospital import HospitalHoloClean
from ml.tools.openrefine.OpenRefine import OpenRefine

#one rule for all columns:
# if(contains(value, "x"), "error", value)
# takes 3 mins to execute

data = HospitalHoloClean()

tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/Hospital/result/hosp_dirty_holoclean_open_refine.tsv",
                  data=data)

print "Fscore: " + str(tool.calculate_total_fscore())
print "Precision: " + str(tool.calculate_total_precision())
print "Recall: " + str(tool.calculate_total_recall())

for c in range(data.shape[1]):
    print tool.calculate_fscore_by_column(c)