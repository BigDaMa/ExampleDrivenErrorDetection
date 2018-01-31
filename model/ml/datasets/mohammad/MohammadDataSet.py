import pandas as pd

from ml.datasets.DataSet import DataSet


class MohammadDataSet(DataSet):
    def __init__(self, name, o, r, p):
        path_to_mohammad_git_repo = "/home/felix/dactor/"

        path_dirty = path_to_mohammad_git_repo + "datasets/" + name + "_o" + str(o) + "_r" + str(r) + "_p" + str(p) + "/" + name + "_o" + str(o) + "_r" + str(r) + "_p" + str(p) + ".csv"
        path_clean = path_to_mohammad_git_repo + "ground-truths/" + name + "/" + name + ".csv"

        dirty_pd = pd.read_csv(path_dirty, header=0, dtype=object, na_filter=False)
        clean_pd = pd.read_csv(path_clean, header=0, dtype=object, na_filter=False)

        super(MohammadDataSet, self).__init__("BlackOak", dirty_pd, clean_pd)

    def validate(self):
        print "validate"
if __name__ == '__main__':

    data = MohammadDataSet("tax", 20, 30, 10)