from sets import Set

from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.tools.nadeef_repair.FD import FD
from ml.tools.nadeef_repair.NadeefAll import NadeefAll

data = BlackOakDataSetUppercase()

rules = []
rules.append(FD(Set(["ZIP"]), "City"))
rules.append(FD(Set(["ZIP"]), "State"))

rules.append(FD(Set(["POCityStateZip"]), "POBox"))
rules.append(FD(Set(["POCityStateZip"]), "City")) #runs out of memory
rules.append(FD(Set(["POCityStateZip"]), "ZIP")) #runs out of memory

'''
# by information gain
rules.append(FD(Set(["ZIP"]), "State"))
rules.append(FD(Set(["Address"]), "State"))

rules.append(FD(Set(["Address","FirstName","POBox"]), "POCityStateZip"))
rules.append(FD(Set(["Address", "City", "FirstName", "POCityStateZip"]), "POBox"))
'''

nadeef = NadeefAll(data, rules)