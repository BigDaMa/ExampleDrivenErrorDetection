from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.tools.openrefine.OpenRefine import OpenRefine

#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/0_upper_case/BlackOak.tsv")
#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/1_ZIP_is_numeric/BlackOak.tsv")
#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/2_State_length_2/BlackOak.tsv")
#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/3_SSN_is_numeric/BlackOak.tsv")
#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/4_ZIP_length_5/BlackOak.tsv")
#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/5_City_not_SAN/BlackOak.tsv")
#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/6_City_not_LOS/BlackOak.tsv")



data = BlackOakDataSetUppercase()
tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/BlackOak/upper_case/BlackOakUppercase_dirty-csv.tsv", data=data)
#tool = OpenRefine("/home/felix/SequentialPatternErrorDetection/OpenRefine/BlackOak/upper_case/BlackOakUppercase_dirty_morerules-csv.tsv", data=data)


print "Fscore: " + str(tool.calculate_total_fscore())
print "Precision: " + str(tool.calculate_total_precision())
print "Recall: " + str(tool.calculate_total_recall())

'''
false_positives_ids = np.where(np.logical_and(tool.matrix_detected[:,10] == True, data.matrix_is_error[:,10] == False))[0]

print false_positives_ids

for i in range(len(false_positives_ids)):
    print "as error detected but clean: " + str(data.dirty_pd.values[false_positives_ids[i],10])
'''


for c in range(data.shape[1]):
    print data.clean_pd.columns[c]
    print "fscore: " + str(tool.calculate_fscore_by_column(c))
    print "recall: " + str(tool.calculate_recall_by_column(c))
    print "precision: " + str(tool.calculate_precision_by_column(c))
    print ""


