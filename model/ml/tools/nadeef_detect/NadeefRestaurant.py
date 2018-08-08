from sets import Set

from ml.datasets.RestaurantMohammad.Restaurant import Restaurant
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect

data = Restaurant()

rules = []


print data.clean_pd.columns

rules.append(FD(Set(["city"]), "state"))
rules.append(FD(Set(["zipcode"]), "state"))

#old
'''
#rules.append(FD(Set(["ZIP"]), "City"))
rules.append(FD(Set(["ZIP"]), "State"))

rules.append(FD(Set(["POCityStateZip"]), "POBox"))
rules.append(FD(Set(["POCityStateZip"]), "City")) #runs out of memory
rules.append(FD(Set(["POCityStateZip"]), "ZIP")) #runs out of memory
'''

'''
# by information gain
rules.append(FD(Set(["ZIP"]), "State"))
rules.append(FD(Set(["Address"]), "State"))

rules.append(FD(Set(["Address","FirstName","POBox"]), "POCityStateZip"))
rules.append(FD(Set(["Address", "City", "FirstName", "POCityStateZip"]), "POBox"))
'''

nadeef = NadeefDetect(data, rules, log_file="/home/felix/ExampleDrivenErrorDetection/log/NADEEF/Restaurant.txt")