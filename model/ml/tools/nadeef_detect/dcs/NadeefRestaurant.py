from sets import Set

from ml.datasets.RestaurantMohammad.Restaurant import Restaurant
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.UDF import UDF
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect

from ml.configuration.Config import Config
import os
import time

path_folder = Config.get("logging.folder") + "/out/nadeef"

if not os.path.exists(path_folder):
    os.makedirs(path_folder)



data = Restaurant()



rules = []

rules.append(FD(Set(["extra_phones", "payment_method", "zipcode"]), "city"))

'''
rules.append(FD(Set(["city"]), "state"))
rules.append(FD(Set(["zipcode"]), "state"))
'''

'''
[restaurant2.csv.extra-phones,
 restaurant2.csv.payment-method,
 restaurant2.csv.zipCode]	restaurant2.csv.city


[restaurant2.csv.aka,
 restaurant2.csv.payment-method,
 restaurant2.csv.zipCode]	restaurant2.csv.city

[restaurant2.csv.payment-method,
 restaurant2.csv.years-in-business,
 restaurant2.csv.zipCode]	restaurant2.csv.city

[restaurant2.csv.streetAddress,
 restaurant2.csv.zipCode]	restaurant2.csv.city
'''




ts = time.time()
log_file = path_folder + "/" + str(data.name) + "_time_" + str(ts) + "_Nadeef.txt"
nadeef = NadeefDetect(data, rules, log_file=log_file)
nadeef.tool.write_detected_matrix()
