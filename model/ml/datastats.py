import numpy as np

from ml.datasets.hospital import HospitalHoloClean

#data = BlackOakDataSet()
data = HospitalHoloClean()
#data = FlightHoloClean()

#data = BlackOakDataSetUppercase()

#data.dirty_pd.to_csv('hosp.csv', index=False)
#data.dirty_pd.to_csv('flight.csv', index=False)

#data.clean_pd.to_csv('blackoak_clean.csv', index=False, sep=',', escapechar='\\', quotechar='\'', na_rep="NA")
#data.clean_pd.to_csv('hosp_clean.csv', index=False, sep=',', escapechar='\\', quotechar='\'', na_rep="NA")
#data.clean_pd.to_csv('flight_clean.csv', index=False, sep=',', escapechar='\\', quotechar='\'', na_rep="NA")

print len(data.clean_pd[data.clean_pd.columns[0]].unique())

print data.shape

print float(np.sum(data.matrix_is_error)) / float(data.shape[0] * data.shape[1])