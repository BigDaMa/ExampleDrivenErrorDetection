from sets import Set

from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.UDF import UDF
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect

from ml.configuration.Config import Config
import os
import time

path_folder = Config.get("logging.folder") + "/out/nadeef"

if not os.path.exists(path_folder):
    os.makedirs(path_folder)

data = BlackOakDataSetUppercase()

rules = []

#only FDs with general coverage = 1.0 that are related to ID by HyFD #check


rules.append(UDF('state', 'value != null && value.length() != 2'))
rules.append(UDF('zip', '(value != null && value.length() != 5)'))
rules.append(UDF('ssn', '(value != null && !isNumeric(value))'))


rules.append(UDF('city', 'value != null && value.equals("SAN")'))
rules.append(UDF('city', 'value != null && value.equals("SANTA")'))
rules.append(UDF('city', 'value != null && value.equals("LOS")'))
rules.append(UDF('city', 'value != null && value.equals("EL")'))
rules.append(UDF('city', 'value != null && value.equals("NORTH")'))
rules.append(UDF('city', 'value != null && value.equals("PALM")'))
rules.append(UDF('city', 'value != null && value.equals("WEST")'))

#rules.append(FD(Set(["ZIP"]), "State"))
#rules.append(FD(Set(["Address"]), "State"))


ts = time.time()
log_file = path_folder + "/" + str(data.name) + "_time_" + str(ts) + "_Nadeef.txt"
nadeef = NadeefDetect(data, rules, log_file=log_file)
nadeef.tool.write_detected_matrix()

