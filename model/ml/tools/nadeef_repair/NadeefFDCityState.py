from sets import Set

from ml.datasets.functional_dependency_data import FD_City_ZIP
from ml.tools.nadeef_repair.FD import FD
from ml.tools.nadeef_repair.NadeefAll import NadeefAll

data = FD_City_ZIP(1000, 0.2, 10000)

rules = []
#rules.append(FD(Set(["zip"]), "city"))
rules.append(FD(Set(["city"]), "zip"))

nadeef = NadeefAll(data, rules)