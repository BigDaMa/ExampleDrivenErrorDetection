from ml.datasets.flights import FlightHoloClean


f = FlightHoloClean()

print len(f.dirty_pd[f.dirty_pd.columns[6]].unique())