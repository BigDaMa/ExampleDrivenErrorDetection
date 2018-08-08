from sets import Set

from ml.datasets.RestaurantMohammad.Restaurant import Restaurant
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.UDF import UDF
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect


#according to FUN and fdmine, no perfect FDs
data = Restaurant()

rules = []


#FDs
#only big FDs that do not bring any benefit
#rules.append(FD(Set(["aka", 'extra_phones', 'name', 'streetAddress', 'website', 'years_in_business']), "payment_method"))








nadeef = NadeefDetect(data, rules, log_file="/home/felix/ExampleDrivenErrorDetection/log/NADEEF/BlackoakUppercase.txt")