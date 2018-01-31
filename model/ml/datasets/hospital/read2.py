import numpy as np

from ml.datasets.hospital import HospitalHoloClean

data = HospitalHoloClean()

print len(data.dirty_pd[data.dirty_pd.columns[1]].unique())


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

X_int = np.matrix(LabelEncoder().fit_transform(data.dirty_pd[data.dirty_pd.columns[1]])).transpose()

# transform to binary
X_bin = OneHotEncoder().fit_transform(X_int)

print X_bin.shape