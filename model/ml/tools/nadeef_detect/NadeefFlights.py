from sets import Set

from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect

data = FlightHoloClean()

rules = []
rules.append(FD(Set(["flight"]), "act_arr_time"))
rules.append(FD(Set(["flight"]), "sched_arr_time"))
rules.append(FD(Set(["flight"]), "act_dep_time"))
rules.append(FD(Set(["flight"]), "sched_dep_time"))


rules.append(FD(Set(["act_arr_time", "sched_arr_time"]), "act_dep_time"))
rules.append(FD(Set(["act_arr_time", "sched_arr_time"]), "sched_dep_time"))

rules.append(FD(Set(["act_arr_time", "act_dep_time"]), "sched_arr_time"))
rules.append(FD(Set(["act_arr_time", "act_dep_time"]), "sched_dep_time"))

rules.append(FD(Set(["act_arr_time", "sched_dep_time"]), "sched_arr_time"))
rules.append(FD(Set(["act_arr_time", "sched_dep_time"]), "act_dep_time"))

rules.append(FD(Set(["act_dep_time", "sched_arr_time"]), "act_arr_time"))
rules.append(FD(Set(["act_dep_time", "sched_arr_time"]), "sched_dep_time"))

rules.append(FD(Set(["sched_arr_time", "sched_dep_time"]), "act_arr_time"))
rules.append(FD(Set(["sched_arr_time", "sched_dep_time"]), "act_dep_time"))

nadeef = NadeefDetect(data, rules, log_file="/home/felix/SequentialPatternErrorDetection/nadeef/log/flights_detect.txt")