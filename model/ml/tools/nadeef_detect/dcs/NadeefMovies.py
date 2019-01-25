from sets import Set

from ml.datasets.MoviesMohammad.Movies import Movies
from ml.tools.nadeef_detect.FD import FD
from ml.tools.nadeef_detect.UDF import UDF
from ml.tools.nadeef_detect.NadeefDetect import NadeefDetect

from ml.configuration.Config import Config
import os
import time

path_folder = Config.get("logging.folder") + "/out/nadeef"

if not os.path.exists(path_folder):
    os.makedirs(path_folder)

# alot of fds
data = Movies()

my_list = list(data.clean_pd.columns)

my_list[7] = 'Cast_1'

data.clean_pd.columns = my_list
data.dirty_pd.columns = my_list


rules = []



'''
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'RatingCount']), "Id"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'RatingCount']), "Name"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'RatingCount']), "Release_Date"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'RatingCount']), "Director"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'RatingCount']), "Genre"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'RatingCount']), "Year"))

rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'Genre']), "Id"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'Genre']), "Name"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'Genre']), "RatingCount"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'Genre']), "Director"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'Genre']), "Year"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'Genre']), "RatingValue"))

rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'RatingValue']), "Id"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'RatingValue']), "Name"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'RatingValue']), "Release_Date"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'RatingValue']), "Director"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'RatingValue']), "Genre"))
rules.append(FD(Set(["Cast_1", 'Creator', 'Description', 'RatingValue']), "Year"))

rules.append(FD(Set(["Cast_1", 'Filming_Locations', 'Description', 'RatingCount', 'Release_Date']), "Id"))
rules.append(FD(Set(["Cast_1", 'Filming_Locations', 'Description', 'RatingCount', 'Release_Date']), "Name"))
rules.append(FD(Set(["Cast_1", 'Filming_Locations', 'Description', 'RatingCount', 'Release_Date']), "Creator"))
rules.append(FD(Set(["Cast_1", 'Filming_Locations', 'Description', 'RatingCount', 'Release_Date']), "Director"))

rules.append(FD(Set(["Cast_1", 'Description', 'Director', 'RatingCount']), "Id"))
rules.append(FD(Set(["Cast_1", 'Description', 'Director', 'RatingCount']), "Name"))

rules.append(FD(Set(["Cast_1", 'Filming_Locations', 'Description', 'RatingCount', 'Year']), "Id"))
rules.append(FD(Set(["Cast_1", 'Filming_Locations', 'Description', 'RatingCount', 'Year']), "Name"))
rules.append(FD(Set(["Cast_1", 'Filming_Locations', 'Description', 'RatingCount', 'Year']), "Genre"))

rules.append(FD(Set(["Cast_1", 'Description', 'Genre', 'RatingCount']), "Id"))
rules.append(FD(Set(["Cast_1", 'Description', 'Genre', 'RatingCount']), "Name"))
rules.append(FD(Set(["Cast_1", 'Description', 'Genre', 'RatingCount']), "Creator"))
rules.append(FD(Set(["Cast_1", 'Description', 'Genre', 'RatingCount']), "Director"))
rules.append(FD(Set(["Cast_1", 'Description', 'Genre', 'RatingCount']), "Filming_Locations"))
rules.append(FD(Set(["Cast_1", 'Description', 'Genre', 'RatingCount']), "Year"))

rules.append(FD(Set(["Cast_1", 'Filming_Locations', 'Description', 'RatingValue', 'Release_Date']), "Id"))
rules.append(FD(Set(["Cast_1", 'Filming_Locations', 'Description', 'RatingValue', 'Release_Date']), "Name"))
rules.append(FD(Set(["Cast_1", 'Filming_Locations', 'Description', 'RatingValue', 'Release_Date']), "Creator"))
rules.append(FD(Set(["Cast_1", 'Filming_Locations', 'Description', 'RatingValue', 'Release_Date']), "Director"))
rules.append(FD(Set(["Cast_1", 'Filming_Locations', 'Description', 'RatingValue', 'Release_Date']), "Genre"))

rules.append(FD(Set(["Cast_1", 'Description', 'Genre', 'Director']), "Id"))
rules.append(FD(Set(["Cast_1", 'Description', 'Genre', 'Director']), "Name"))
rules.append(FD(Set(["Cast_1", 'Description', 'Genre', 'Director']), "RatingCount"))
rules.append(FD(Set(["Cast_1", 'Description', 'Genre', 'Director']), "RatingValue"))


rules.append(FD(Set(["Cast_1", 'Description', 'RatingValue', 'Director']), "Id"))
rules.append(FD(Set(["Cast_1", 'Description', 'RatingValue', 'Director']), "Name"))
'''


'''
        # duration: not "hr"
        # Genre: not "&"
        Release Date
        '''



rules.append(UDF('Year', 'value != null && value.length() != 4'))
rules.append(UDF('RatingValue', 'value != null && value.length() != 3'))
rules.append(UDF('Id', 'value != null && value.length() != 9'))
rules.append(UDF('Duration', 'value != null && value.length() > 7'))


#rules.append(FD(Set(["Cast", "Duration"]), "Actors")) #0
#rules.append(FD(Set(["Description", "Release_Date"]), "Country"))
#rules.append(FD(Set(["Name", "Year"]), "Language"))


#FDs with best info gain did not bring anything


#FDs
#only big FDs that do not bring any benefit
#rules.append(FD(Set(["aka", 'extra_phones', 'name', 'streetAddress', 'website', 'years_in_business']), "payment_method"))




ts = time.time()
log_file = path_folder + "/" + str(data.name) + "_time_" + str(ts) + "_Nadeef.txt"
nadeef = NadeefDetect(data, rules, log_file=log_file)

nadeef.tool.write_detected_matrix()