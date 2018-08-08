from sets import Set

from ml.datasets.MoviesMohammad.Movies import Movies
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.UDF import UDF
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect


#according to FUN and fdmine, no perfect FDs
data = Movies()

my_list = list(data.clean_pd.columns)

my_list[7] = 'Cast_1'

data.clean_pd.columns = my_list
data.dirty_pd.columns = my_list


rules = []

'''
        # duration: not "hr"
        # Genre: not "&"
        Release Date
        '''

rules.append(UDF('Year', 'value != null && value.length() != 4'))
rules.append(UDF('RatingValue', 'value != null && value.length() != 3'))
rules.append(UDF('Id', 'value != null && value.length() != 9'))
rules.append(UDF('Duration', 'value != null && value.length() > 7'))

#FDs
#only big FDs that do not bring any benefit
#rules.append(FD(Set(["aka", 'extra_phones', 'name', 'streetAddress', 'website', 'years_in_business']), "payment_method"))








nadeef = NadeefDetect(data, rules, log_file="/home/felix/ExampleDrivenErrorDetection/log/NADEEF/BlackoakUppercase.txt")