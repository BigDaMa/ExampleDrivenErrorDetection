from ml.datasets.mohammad import MohammadDataSet
from ml.tools.openrefine.OpenRefine import OpenRefine

#one rule for all columns:
# if(contains(value, "x"), "error", value)
# takes 3 mins to execute

data = MohammadDataSet("tax", 20, 30, 10)

tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/tax/result/tax_o20_r30_p10-csv-with-minus-rule.tsv",
                  data=data)

print "Fscore: " + str(tool.calculate_total_fscore())
print "Precision: " + str(tool.calculate_total_precision())
print "Recall: " + str(tool.calculate_total_recall())

for c in range(data.shape[1]):
    print tool.calculate_fscore_by_column(c)