from sets import Set

from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.UDF import UDF
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect

data = BlackOakDataSetUppercase()

rules = []

#only FDs with general coverage = 1.0 that are related to ID

'''
rules.append(UDF('state', 'value != null && value.length() != 2'))
rules.append(UDF('zip', '(value != null && value.length() != 5)'))
rules.append(UDF('ssn', '(value != null && !isNumeric(value))'))


rules.append(UDF('city', 'value != null && value.equals("SAN")'))
rules.append(UDF('city', 'value != null && value.equals("SANTA")'))
rules.append(UDF('city', 'value != null && value.equals("LOS")'))
rules.append(UDF('city', 'value != null && value.equals("EL")'))
rules.append(UDF('city', 'value != null && value.equals("NORTH")'))
rules.append(UDF('city', 'value != null && value.equals("PALM")'))
rules.append(UDF('city', 'value != null && value.equals("WEST")'))
'''




nadeef = NadeefDetect(data, rules, log_file="/home/felix/ExampleDrivenErrorDetection/log/NADEEF/BlackoakUppercase.txt")