from sets import Set

from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.UDF import UDF
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect

from ml.configuration.Config import Config
import os
import time

path_folder = Config.get("logging.folder") + "/out/nadeef"

if not os.path.exists(path_folder):
    os.makedirs(path_folder)


# alot of fds by dfd
data = HospitalHoloClean()

rules = []





rules.append(UDF('provider_number', '(value != null && !isNumeric(value))'))
rules.append(UDF('zip_code', '(value != null && !isNumeric(value))'))
rules.append(UDF('phone_number', '(value != null && !isNumeric(value))'))
rules.append(UDF('emergency_service', '!(value.equals("Yes") || value.equals("No"))'))
rules.append(UDF('state', '!(value.equals("AL") || value.equals("AK"))'))



#rules.append(FD(Set(["phone_number"]), "zip_code"))
#rules.append(FD(Set(["phone_number"]), "city"))
#rules.append(FD(Set(["phone_number"]), "state"))

#rules.append(FD(Set(["zip_code"]), "city"))
#rules.append(FD(Set(["zip_code"]), "state"))

#rules.append(FD(Set(["measure_code"]), "measure_name"))
#rules.append(FD(Set(["measure_code"]), "condition"))

#rules.append(FD(Set(["measure_code", "provider_number"]), "stateavg"))
#rules.append(FD(Set(["measure_code", "state"]), "stateavg"))







ts = time.time()
log_file = path_folder + "/" + str(data.name) + "_time_" + str(ts) + "_Nadeef.txt"
nadeef = NadeefDetect(data, rules, log_file=log_file)
nadeef.tool.write_detected_matrix('/tmp/matrix_detected_hospital_nadeef.npy')