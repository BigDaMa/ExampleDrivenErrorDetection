import pandas as pd
from ml.datasets.DataSet import DataSet
from ml.configuration.Config import Config


class BlackOakDataSetUppercase(DataSet):
    name = "BlackOakUppercase"

    def __init__(self, duplicate_factor=1):
        path_to_dirty = Config.get("blackoak.data") + "/inputDB.csv"
        path_to_clean = Config.get("blackoak.data") + "/groundDB.csv"

        dirty_pd_init = pd.read_csv(path_to_dirty, header=0, dtype=object, na_filter=False)
        clean_pd_init = pd.read_csv(path_to_clean, header=0, dtype=object, na_filter=False)

        #print dirty_pd_init.dtypes
        #print clean_pd_init.dtypes

        dirty_pd = self.uppercase(dirty_pd_init)
        clean_pd = self.uppercase(clean_pd_init)

        #dirty_pd.to_csv("BlackOakUppercase_dirty_new.csv", index=False)

        duplicated_clean = clean_pd.copy(deep=True)
        duplicated_dirty = dirty_pd.copy(deep=True)

        for i in range(duplicate_factor - 1):
            copy_dirty = dirty_pd.copy(deep=True)
            copy_clean = clean_pd.copy(deep=True)

            duplicated_dirty = duplicated_dirty.append(copy_dirty, ignore_index=True)
            duplicated_clean = duplicated_clean.append(copy_clean, ignore_index=True)


        super(BlackOakDataSetUppercase, self).__init__(BlackOakDataSetUppercase.name, duplicated_dirty, duplicated_clean)

    def validate(self):
        print "validate"


    def uppercase(self, df):
        df_new = df.copy(deep=True)
        capitalizer = lambda x: str(x).upper()

        for col in df.columns:
            df_new[col] = df_new[col].apply(capitalizer)

        return df_new

if __name__ == '__main__':

    data = BlackOakDataSetUppercase()