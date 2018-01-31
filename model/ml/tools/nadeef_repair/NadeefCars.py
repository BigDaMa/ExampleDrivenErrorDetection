from sets import Set

from ml.datasets.mohammad import MohammadDataSet
from ml.tools.nadeef_repair.FD import FD
from ml.tools.nadeef_repair.NadeefAll import NadeefAll

data = MohammadDataSet("cars", 30, 20, 20)

rules = []
rules.append(FD(Set(["title_varchar"]), "brand_name_varchar"))

nadeef = NadeefAll(data, rules)