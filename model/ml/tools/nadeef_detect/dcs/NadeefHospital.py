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


#rules.append(FD(Set(["city"]), "state"))

#FDs
'''
#1 general coberage FDs do not help at all
rules.append(FD(Set(["phone_number", 'measure_name']), "score"))
rules.append(FD(Set(["phone_number", 'measure_name']), "sample"))
rules.append(FD(Set(["phone_number", 'measure_name']), "stateavg"))
rules.append(FD(Set(["address1", 'stateavg']), "score"))
rules.append(FD(Set(["address1", 'stateavg']), "sample"))
rules.append(FD(Set(["hospital_name", 'stateavg']), "score"))
rules.append(FD(Set(["hospital_name", 'stateavg']), "sample"))
rules.append(FD(Set(["provider_number", 'stateavg']), "score"))
rules.append(FD(Set(["provider_number", 'stateavg']), "sample"))


rules.append(FD(Set(["city", 'hospital_owner', 'sample', 'stateavg']), "provider_number"))
rules.append(FD(Set(["city", 'hospital_owner', 'sample', 'stateavg']), "hospital_name"))
rules.append(FD(Set(["city", 'hospital_owner', 'sample', 'stateavg']), "address1"))
rules.append(FD(Set(["city", 'hospital_owner', 'sample', 'stateavg']), "zip_code"))
rules.append(FD(Set(["city", 'hospital_owner', 'sample', 'stateavg']), "phone_number"))
rules.append(FD(Set(["city", 'hospital_owner', 'sample', 'stateavg']), "score"))

rules.append(FD(Set(["address1", 'measure_name']), "score"))
rules.append(FD(Set(["address1", 'measure_name']), "sample"))

rules.append(FD(Set(["hospital_name", 'measure_name']), "score"))
rules.append(FD(Set(["hospital_name", 'measure_name']), "sample"))
rules.append(FD(Set(["hospital_name", 'measure_name']), "stateavg"))

rules.append(FD(Set(["provider_number", 'measure_name']), "score"))
rules.append(FD(Set(["provider_number", 'measure_name']), "sample"))
rules.append(FD(Set(["provider_number", 'measure_name']), "stateavg"))

rules.append(FD(Set(["sample", 'measure_name', 'zip_code']), "score"))

rules.append(FD(Set(["city", 'hospital_owner', 'sample', 'measure_name']), "provider_number"))
rules.append(FD(Set(["city", 'hospital_owner', 'sample', 'measure_name']), "hospital_name"))
rules.append(FD(Set(["city", 'hospital_owner', 'sample', 'measure_name']), "address1"))
rules.append(FD(Set(["city", 'hospital_owner', 'sample', 'measure_name']), "zip_code"))
rules.append(FD(Set(["city", 'hospital_owner', 'sample', 'measure_name']), "phone_number"))
rules.append(FD(Set(["city", 'hospital_owner', 'sample', 'measure_name']), "score"))

rules.append(FD(Set(["city", 'hospital_owner', 'zip_code', 'measure_code']), "score"))
rules.append(FD(Set(["city", 'hospital_owner', 'zip_code', 'measure_code']), "sample"))

rules.append(FD(Set(["city", 'sample', 'zip_code', 'stateavg']), "score"))

rules.append(FD(Set(['measure_code', 'phone_number']), "score"))
rules.append(FD(Set(['measure_code', 'phone_number']), "sample"))
rules.append(FD(Set(['measure_code', 'phone_number']), "stateavg"))

rules.append(FD(Set(['hospital_owner', 'stateavg', 'zip_code']), "score"))
rules.append(FD(Set(['hospital_owner', 'stateavg', 'zip_code']), "sample"))

rules.append(FD(Set(['hospital_owner', 'measure_name', 'zip_code']), "score"))
rules.append(FD(Set(['hospital_owner', 'measure_name', 'zip_code']), "sample"))

rules.append(FD(Set(['stateavg', 'phone_number']), "score"))
rules.append(FD(Set(['stateavg', 'phone_number']), "sample"))

rules.append(FD(Set(["address1", 'measure_code']), "score"))
rules.append(FD(Set(["address1", 'measure_code']), "sample"))
rules.append(FD(Set(["address1", 'measure_code']), "stateavg"))

rules.append(FD(Set(["hospital_name", 'measure_code']), "score"))
rules.append(FD(Set(["hospital_name", 'measure_code']), "sample"))
rules.append(FD(Set(["hospital_name", 'measure_code']), "stateavg"))

rules.append(FD(Set(["provider_number", 'measure_code']), "score"))
rules.append(FD(Set(["provider_number", 'measure_code']), "sample"))
rules.append(FD(Set(["provider_number", 'measure_code']), "stateavg"))


rules.append(FD(Set(["sample", 'measure_code', 'zip_code']), "score"))

rules.append(FD(Set(["city", 'hospital_owner', 'measure_code', 'sample']), "provider_number"))
rules.append(FD(Set(["city", 'hospital_owner', 'measure_code', 'sample']), "hospital_name"))
rules.append(FD(Set(["city", 'hospital_owner', 'measure_code', 'sample']), "address1"))
rules.append(FD(Set(["city", 'hospital_owner', 'measure_code', 'sample']), "zip_code"))
rules.append(FD(Set(["city", 'hospital_owner', 'measure_code', 'sample']), "phone_number"))
rules.append(FD(Set(["city", 'hospital_owner', 'measure_code', 'sample']), "score"))
'''



rules.append(UDF('provider_number', '(value != null && !isNumeric(value))'))
rules.append(UDF('zip_code', '(value != null && !isNumeric(value))'))
rules.append(UDF('phone_number', '(value != null && !isNumeric(value))'))
rules.append(UDF('emergency_service', '!(value.equals("Yes") || value.equals("No"))'))
rules.append(UDF('state', '!(value.equals("AL") || value.equals("AK"))'))




ts = time.time()
log_file = path_folder + "/" + str(data.name) + "_time_" + str(ts) + "_Nadeef.txt"
nadeef = NadeefDetect(data, rules, log_file=log_file)