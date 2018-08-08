from sets import Set

from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.UDF import UDF
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect


#according to FUN and fdmine, no perfect FDs
data = HospitalHoloClean()

rules = []


#FDs
#rules.append(FD(Set(["phone_number", 'measure_name']), "score"))
#rules.append(FD(Set(["phone_number", 'measure_name']), "sample"))
#rules.append(FD(Set(["phone_number", 'measure_name']), "stateavg"))
#rules.append(FD(Set(["address1", 'stateavg']), "score"))
#rules.append(FD(Set(["address1", 'stateavg']), "sample"))
#rules.append(FD(Set(["hospital_name", 'stateavg']), "score"))
#rules.append(FD(Set(["hospital_name", 'stateavg']), "sample"))
#rules.append(FD(Set(["city", 'hospital_owner', 'sample', 'stateavg']), "provider_number"))
#rules.append(FD(Set(["city", 'hospital_owner', 'sample', 'stateavg']), "zip_code"))
#rules.append(FD(Set(["stateavg"]), 'county_name'))

rules.append(UDF('provider_number', '(value != null && !isNumeric(value))'))
rules.append(UDF('zip_code', '(value != null && !isNumeric(value))'))
rules.append(UDF('phone_number', '(value != null && !isNumeric(value))'))
rules.append(UDF('emergency_service', '!(value.equals("Yes") || value.equals("No"))'))
rules.append(UDF('state', '!(value.equals("AL") || value.equals("AK"))'))





nadeef = NadeefDetect(data, rules, log_file="/home/felix/ExampleDrivenErrorDetection/log/NADEEF/BlackoakUppercase.txt")