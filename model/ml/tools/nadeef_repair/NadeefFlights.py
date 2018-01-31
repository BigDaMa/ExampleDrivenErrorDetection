from sets import Set

from ml.datasets.flights import FlightHoloClean
from ml.tools.nadeef_repair.FD import FD
from ml.tools.nadeef_repair.NadeefAll import NadeefAll

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

nadeef = NadeefAll(data, rules, log_file="/home/felix/SequentialPatternErrorDetection/nadeef_repair/log/flights.txt")