from sets import Set

from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.UDF import UDF
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect

from ml.configuration.Config import Config
import os
import time

path_folder = Config.get("logging.folder") + "/out/nadeef"

if not os.path.exists(path_folder):
    os.makedirs(path_folder)

data = FlightHoloClean()

rules = []


#no FDs with general coverage = 1.0 # HyFD-1.1 #check


rules.append(UDF('sched_dep_time', 'value == null || (value != null && value.length() > 10)'))
rules.append(UDF('act_dep_time', 'value == null || (value != null && value.length() > 10)'))
rules.append(UDF('sched_arr_time', 'value == null || (value != null && value.length() > 10)'))
rules.append(UDF('act_arr_time', 'value == null || (value != null && value.length() > 10)'))


#rules.append(FD(Set(["flight"]), "act_arr_time"))
#rules.append(FD(Set(["flight"]), "sched_arr_time"))
#rules.append(FD(Set(["flight"]), "act_dep_time"))
#rules.append(FD(Set(["flight"]), "sched_dep_time"))

#rules.append(FD(Set(["act_arr_time", "sched_arr_time"]), "act_dep_time"))
#rules.append(FD(Set(["act_arr_time", "sched_arr_time"]), "sched_dep_time"))

#rules.append(FD(Set(["act_arr_time", "act_dep_time"]), "sched_arr_time"))
#rules.append(FD(Set(["act_arr_time", "act_dep_time"]), "sched_dep_time"))

#rules.append(FD(Set(["act_arr_time", "sched_dep_time"]), "sched_arr_time"))
#rules.append(FD(Set(["act_arr_time", "sched_dep_time"]), "act_dep_time"))


#rules.append(FD(Set(["act_dep_time", "sched_arr_time"]), "act_arr_time"))
#rules.append(FD(Set(["act_dep_time", "sched_arr_time"]), "sched_dep_time"))

#rules.append(FD(Set(["sched_arr_time", "sched_dep_time"]), "act_arr_time"))

#works
rules.append(FD(Set(["sched_arr_time", "sched_dep_time"]), "act_dep_time"))



ts = time.time()
log_file = path_folder + "/" + str(data.name) + "_time_" + str(ts) + "_Nadeef.txt"
nadeef = NadeefDetect(data, rules, log_file=log_file)
nadeef.tool.write_detected_matrix()