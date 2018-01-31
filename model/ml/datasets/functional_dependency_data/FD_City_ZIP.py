import numpy as np
import pandas as pd

from ml.datasets.DataSet import DataSet


class FD_City_ZIP(DataSet):

    def __init__(self, N, error_probability, domain_size, seed=42):
        np.random.seed(seed=seed)

        path_to_zip_dict = "/home/felix/DataIntegration/DataIntegration/Task1/auxiliary/free-zipcode-database.csv"

        zipDatabase = pd.read_csv(path_to_zip_dict, header=0, dtype=object)

        domain_size = min(zipDatabase.shape[0], domain_size)

        tuple_ids = np.random.randint(0, domain_size, N)
        is_error = np.random.rand(N) < error_probability

        #print tuple_ids
        t = zipDatabase.values[tuple_ids]

        selected = t[:,[1,3]]

        #print selected

        #print selected
        error_ids = np.where(is_error)[0]

        #print error_ids

        clean_pd = pd.DataFrame(selected, columns=['zip', 'city'], dtype=object)

        to_be_dirty = np.copy(selected)

        #change ZIP to another real ZIP
        for error_id in error_ids:
            to_be_dirty[error_id,0] = zipDatabase.values[np.random.randint(0, domain_size),1]

        dirty_pd = pd.DataFrame(to_be_dirty, columns=['zip', 'city'], dtype=object)

        #print dirty_pd

        super(FD_City_ZIP, self).__init__("FD_City_ZIP", dirty_pd, clean_pd)




    def validate(self):
        print "validate"

if __name__ == '__main__':

    data = FD_City_ZIP()