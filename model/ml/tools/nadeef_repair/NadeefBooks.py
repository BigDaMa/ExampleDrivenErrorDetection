from sets import Set

from ml.datasets.mohammad import MohammadDataSet
from ml.tools.nadeef_repair.FD import FD
from ml.tools.nadeef_repair.NadeefAll import NadeefAll

data = MohammadDataSet("books", 30, 30, 10)

rules = []

#'''
#Mohammad's rule
rules.append(FD(Set(["first_author_varchar"]), "language_varchar"))
#'''

#rules.append(FD(Set(["first_author_varchar", "publish_date_varchar", "rating_varchar"]), "language_varchar"))
rules.append(FD(Set(["isbn13_varchar", "publisher_varchar", "rating_varchar", "title_varchar"]), "first_author_varchar"))
rules.append(FD(Set(["description_varchar", "first_author_varchar", "format_varchar", "title_varchar"]), "isbn13_varchar"))





nadeef = NadeefAll(data, rules)