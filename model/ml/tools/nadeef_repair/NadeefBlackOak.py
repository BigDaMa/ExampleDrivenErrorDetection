from sets import Set

from ml.datasets.blackOak import BlackOakDataSet
from ml.tools.nadeef_repair.FD import FD
from ml.tools.nadeef_repair.NadeefAll import NadeefAll

data = BlackOakDataSet()

rules = []
rules.append(FD(Set(["ZIP"]), "City"))
rules.append(FD(Set(["ZIP"]), "State"))

rules.append(FD(Set(["POCityStateZip"]), "POBox"))
#rules.append(FD(Set(["POCityStateZip"]), "City")) #runs out of memory
#rules.append(FD(Set(["POCityStateZip"]), "ZIP")) #runs out of memory

nadeef = NadeefAll(data, rules)