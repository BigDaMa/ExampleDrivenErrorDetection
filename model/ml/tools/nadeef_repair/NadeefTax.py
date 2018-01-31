from sets import Set

from ml.datasets.mohammad import MohammadDataSet
from ml.tools.nadeef_repair.FD import FD
from ml.tools.nadeef_repair.NadeefAll import NadeefAll

data = MohammadDataSet("tax", 20, 30, 10)

rules = []
#rules.append(FD(Set(["fname"]), "gender"))
#rules.append(FD(Set(["city"]), "state"))

#by information gain
'''
#rules.append(FD(Set(["areacode", "maritalstatus"]), "singleexemp"))
#rules.append(FD(Set(["areacode", "maritalstatus"]), "marriedexemp"))
#rules.append(FD(Set(["areacode", "haschild"]), "childexemp"))
#rules.append(FD(Set(["areacode", "salary"]), "rate"))
rules.append(FD(Set(["zip"]), "city"))
rules.append(FD(Set(["zip"]), "state"))
rules.append(FD(Set(["fname"]), "gender"))
rules.append(FD(Set(["phone", "zip"]), "fname"))
#rules.append(FD(Set(["phone", "zip"]), "lname"))
rules.append(FD(Set(["phone", "zip"]), "areacode"))
#rules.append(FD(Set(["phone", "zip"]), "salary"))
#rules.append(FD(Set(["phone", "zip"]), "maritalstatus"))
#rules.append(FD(Set(["phone", "zip"]), "haschild"))
rules.append(FD(Set(["city", "phone"]), "zip"))
rules.append(FD(Set(["fname", "lname", "zip"]), "phone"))
'''

rules.append(FD(Set(["areacode", "maritalstatus"]), "singleexemp"))
rules.append(FD(Set(["areacode", "maritalstatus"]), "marriedexemp"))
rules.append(FD(Set(["areacode", "haschild"]), "childexemp"))
rules.append(FD(Set(["areacode", "salary"]), "rate"))
rules.append(FD(Set(["zip"]), "city"))
rules.append(FD(Set(["zip"]), "state"))
rules.append(FD(Set(["fname"]), "gender"))
rules.append(FD(Set(["phone", "zip"]), "fname"))
rules.append(FD(Set(["phone", "zip"]), "lname"))
rules.append(FD(Set(["phone", "zip"]), "areacode"))
rules.append(FD(Set(["phone", "zip"]), "salary"))
rules.append(FD(Set(["phone", "zip"]), "maritalstatus"))
rules.append(FD(Set(["phone", "zip"]), "haschild"))
rules.append(FD(Set(["city", "phone"]), "zip"))
rules.append(FD(Set(["fname", "lname", "zip"]), "phone"))



nadeef = NadeefAll(data, rules)