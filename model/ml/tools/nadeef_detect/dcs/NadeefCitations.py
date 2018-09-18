from sets import Set

from ml.datasets.Citations.Citation import Citation
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.UDF import UDF
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect

from ml.configuration.Config import Config
import os
import time

path_folder = Config.get("logging.folder") + "/out/nadeef"

if not os.path.exists(path_folder):
    os.makedirs(path_folder)


# according to FUN and fdmine, no perfect FDs
# HyFD only finds ID columns that are involved into FDs
data = Citation()


rules = []




rules.append(UDF('article_jissue', 'value == null'))
rules.append(UDF('article_jvolumn', 'value == null'))
#rules.append(UDF('author_list', 'value == null')) # does not work

#rules.append(FD(Set(['jounral_abbreviation']), 'journal_title'))
rules.append(FD(Set(['jounral_abbreviation']), 'journal_issn'))



#FDs
#only big FDs that do not bring any benefit
#rules.append(FD(Set(["aka", 'extra_phones', 'name', 'streetAddress', 'website', 'years_in_business']), "payment_method"))


ts = time.time()
log_file = path_folder + "/" + str(data.name) + "_time_" + str(ts) + "_Nadeef.txt"
nadeef = NadeefDetect(data, rules, log_file=log_file)