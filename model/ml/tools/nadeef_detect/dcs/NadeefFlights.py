from sets import Set

from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.UDF import UDF
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect

data = FlightHoloClean()

rules = []


#no FDs with general coverage = 1.0


rules.append(UDF('sched_dep_time', 'value == null || (value != null && value.length() > 10)'))
rules.append(UDF('act_dep_time', 'value == null || (value != null && value.length() > 10)'))
rules.append(UDF('sched_arr_time', 'value == null || (value != null && value.length() > 10)'))
rules.append(UDF('act_arr_time', 'value == null || (value != null && value.length() > 10)'))





nadeef = NadeefDetect(data, rules, log_file="/home/felix/ExampleDrivenErrorDetection/log/NADEEF/BlackoakUppercase.txt")