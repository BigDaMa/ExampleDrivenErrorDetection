import numpy as np

from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase

data = BlackOakDataSetUppercase()

sample_size = 14

for c in range(data.shape[1]):
    error_ids = np.where(data.matrix_is_error[:,c])[0]
    print data.clean_pd.columns[c]
    print "number of errors: " + str(np.sum(data.matrix_is_error[:,c]))
    if (len(error_ids) >= sample_size):
        for i in range(sample_size):
            print "dirty: " + str(data.dirty_pd.values[error_ids[i],c]) + " -> clean: " + str(data.clean_pd.values[error_ids[i],c])
    print ""