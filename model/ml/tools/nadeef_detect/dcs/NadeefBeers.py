from sets import Set

from ml.datasets.BeersMohammad.Beers import Beers
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.UDF import UDF
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect

from ml.configuration.Config import Config
import os
import time

path_folder = Config.get("logging.folder") + "/out/nadeef"

if not os.path.exists(path_folder):
    os.makedirs(path_folder)


#according to FUN and fdmine, no perfect FDs
# according to HyFD only ID columns are involved into FDs #check
data = Beers()

my_list = list(data.clean_pd.columns)
my_list[0] = 'anid'
data.clean_pd.columns=my_list
data.dirty_pd.columns=my_list

rules = []

#rules.append(UDF('ounces', 'value.length() > 4'))


rules.append(UDF('ibu', 'value.equals("N/A")'))
rules.append(UDF('abv', '(value != null && !isNumeric(value))'))
rules.append(UDF('city', '((String)tuple.get("state") == null)'))
rules.append(UDF('state', '(value == null)'))


#FDs
#only big FDs that do not bring any benefit
#rules.append(FD(Set(["aka", 'extra_phones', 'name', 'streetAddress', 'website', 'years_in_business']), "payment_method"))


ts = time.time()
log_file = path_folder + "/" + str(data.name) + "_time_" + str(ts) + "_Nadeef.txt"
nadeef = NadeefDetect(data, rules, log_file=log_file)
nadeef.tool.write_detected_matrix()