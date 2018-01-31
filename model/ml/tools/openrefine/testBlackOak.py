from ml.tools.openrefine.OpenRefine import OpenRefine

#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/0_upper_case/BlackOak.tsv")
#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/1_ZIP_is_numeric/BlackOak.tsv")
#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/2_State_length_2/BlackOak.tsv")
#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/3_SSN_is_numeric/BlackOak.tsv")
#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/4_ZIP_length_5/BlackOak.tsv")
#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/5_City_not_SAN/BlackOak.tsv")
#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/6_City_not_LOS/BlackOak.tsv")

tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/all/BlackOak.tsv")

print "Fscore: " + str(tool.calculate_total_fscore())
print "Precision: " + str(tool.calculate_total_precision())
print "Recall: " + str(tool.calculate_total_recall())

for c in range(12):
    print tool.calculate_fscore_by_column(c)