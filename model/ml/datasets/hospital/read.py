from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np


hospital_data = pd.read_csv("/home/felix/NADEEF/examples/hospital500k.csv", sep=',', header=0)
audit_table = pd.read_csv("/home/felix/SequentialPatternErrorDetection/nadeef_repair/audit_big.csv", sep='\t', header=None)


map = {}
map['city'] = 4
map['state'] = 5
map['zipcode'] = 6

def name_to_id(name):
    return map[name]

audit_table[audit_table.columns[4]] = audit_table[audit_table.columns[4]].apply(name_to_id)

print audit_table


matrix_is_error = np.zeros((hospital_data.shape), dtype=bool)

for i in range(len(audit_table)):
    row = audit_table.values[i,2]
    column = audit_table.values[i,4]

    #print str(row) + " " + str(column)

    matrix_is_error[row][column] = True

print np.sum(matrix_is_error)

f = open('/home/felix/SequentialPatternErrorDetection/nadeef_repair/matrix_audit/matrix_is_error', 'w')
np.save(f, matrix_is_error)

f.seek(0)
print np.load(f)