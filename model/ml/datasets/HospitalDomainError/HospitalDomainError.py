from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
import numpy as np
import pandas as pd
from ml.datasets.DataSet import DataSet


class HospitalDomainError(DataSet):
    name = "HospitalDomainError"

    def __init__(self):
        holoclean = HospitalHoloClean()

        rng = np.random.RandomState(42)

        clean_pd = holoclean.clean_pd.copy()
        dirty_pd = holoclean.clean_pd.copy()
        is_error = holoclean.matrix_is_error

        dirty_matrix = dirty_pd.values

        for c in range(clean_pd.shape[1]):
            domain = clean_pd[clean_pd.columns[c]].unique()
            if len(domain) > 1:
                for r in range(clean_pd.shape[0]):
                    if is_error[r, c]:
                        val = dirty_matrix[r, c]
                        while dirty_matrix[r, c] == val:
                            val = domain[rng.randint(len(domain))]

                        print str(dirty_matrix[r, c]) + " -> " + str(val)
                        dirty_matrix[r, c] = val

        dirty_pd = pd.DataFrame(dirty_matrix, columns=holoclean.dirty_pd.columns)

        super(HospitalDomainError, self).__init__(HospitalDomainError.name, dirty_pd, clean_pd)


if __name__ == '__main__':
    data = HospitalDomainError()

    print data.dirty_pd