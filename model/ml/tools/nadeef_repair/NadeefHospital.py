from sets import Set

from ml.datasets.hospital import HospitalHoloClean
from ml.tools.nadeef_repair.FD import FD
from ml.tools.nadeef_repair.NadeefAll import NadeefAll

data = HospitalHoloClean()

rules = []
rules.append(FD(Set(["phone_number"]), "zip_code"))
rules.append(FD(Set(["phone_number"]), "city"))
rules.append(FD(Set(["phone_number"]), "state"))

rules.append(FD(Set(["zip_code"]), "city"))
rules.append(FD(Set(["zip_code"]), "state"))

rules.append(FD(Set(["measure_code"]), "measure_name"))
rules.append(FD(Set(["measure_code"]), "condition"))
rules.append(FD(Set(["measure_code", "provider_number"]), "stateavg"))
rules.append(FD(Set(["measure_code", "state"]), "stateavg"))

nadeef = NadeefAll(data, rules, log_file="/home/felix/SequentialPatternErrorDetection/nadeef_repair/log/hospital.txt")