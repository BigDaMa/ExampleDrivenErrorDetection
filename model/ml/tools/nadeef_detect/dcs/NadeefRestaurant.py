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


# a lot of fds
data = Restaurant()

rules = []


#not paper
rules.append(UDF('categories', 'value != null && !value.contains("Restaurants")'))
#rules.append(UDF('ratingValue', 'value != null && value.length() > 1'))


#FDs
'''
rules.append(FD(Set(["aka", 'extra_phones', 'name', 'streetAddress', 'website', 'phone', 'ratingValue']), "years_in_business"))
rules.append(FD(Set(["aka", 'extra_phones', 'name', 'streetAddress', 'website', 'phone', 'priceRange']), "years_in_business"))
rules.append(FD(Set(["aka", 'extra_phones', 'name', 'streetAddress', 'website', 'phone', 'categories']), "years_in_business"))

rules.append(FD(Set(["aka", 'name', 'streetAddress', 'neighborhood', 'phone', 'categories']), "extra_phones"))
'''

#rules.append(FD(Set(['city']), "state"))
#rules.append(FD(Set(['zipCode']), "state"))




ts = time.time()
log_file = path_folder + "/" + str(data.name) + "_time_" + str(ts) + "_Nadeef.txt"
nadeef = NadeefDetect(data, rules, log_file=log_file)