from sets import Set

from ml.datasets.Citations.Citation import Citation
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.UDF import UDF
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect


#according to FUN and fdmine, no perfect FDs
data = Citation()


rules = []

#rules.append(FD(Set(['article_title']), 'article_languange'))
rules.append(FD(Set(['article_title']), 'article_jcreated_at'))
rules.append(FD(Set(['article_title']), 'author_list'))

'''
rules.append(UDF('Year', 'value != null && value.length() != 4'))
rules.append(UDF('RatingValue', 'value != null && value.length() != 3'))
rules.append(UDF('Id', 'value != null && value.length() != 9'))
rules.append(UDF('Duration', 'value != null && value.length() > 7'))
'''

#FDs
#only big FDs that do not bring any benefit
#rules.append(FD(Set(["aka", 'extra_phones', 'name', 'streetAddress', 'website', 'years_in_business']), "payment_method"))








nadeef = NadeefDetect(data, rules, log_file="/home/felix/ExampleDrivenErrorDetection/log/NADEEF/BlackoakUppercase.txt")