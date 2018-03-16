from sets import Set

from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect

from ml.datasets.BartDataset.BartDataSet import BartDataset
data = BartDataset(BlackOakDataSetUppercase(), "CityFD_20percent")

rules = []

rules.append(FD(Set(["ZIP"]), "City"))

nadeef = NadeefDetect(data, rules, log_file="/home/felix/SequentialPatternErrorDetection/nadeef/log/Bart.txt")