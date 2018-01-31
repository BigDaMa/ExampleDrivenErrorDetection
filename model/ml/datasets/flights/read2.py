from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np


def fillna_df(df):
    for i in range(df.shape[1]):
        if df[df.columns[i]].dtypes.name == "object":
            df[df.columns[i]] = df[df.columns[i]].fillna('')
        else:
            raise Exception('not implemented')
            # todo if numeric
    return df

dirty = pd.read_csv("/home/felix/SequentialPatternErrorDetection/flight/data/clean_flight/2011-12-01-data.txt", sep='\t', header=None)

truth = pd.read_csv("/home/felix/SequentialPatternErrorDetection/flight/data/flight_truth/2011-12-01-truth.txt", sep='\t', header=None)


dirty = fillna_df(dirty)
truth = fillna_df(truth)

print dirty.columns
print truth.columns

dirty = dirty.rename(index=str, columns={0: "Source",
                                         1: "Flight#",
                                         2: "Scheduled departure",
                                         3: "Actual departure",
                                         4: "Departure gate",
                                         5: "Scheduled arrival",
                                         6: "Actual arrival",
                                         7:	"Arrival gate"})
truth = truth.rename(index=str, columns={0: "Flight#",
                                         1: "Scheduled departure",
                                         2: "Actual departure",
                                         3: "Departure gate",
                                         4: "Scheduled arrival",
                                         5: "Actual arrival",
                                         6: "Arrival gate"})

print truth.shape
print len(truth["Flight#"].unique())

print dirty.shape
print len(dirty["Flight#"].unique())

result = pd.merge(dirty, truth, on='Flight#')

print result.shape
print len(result["Flight#"].unique())

res = dirty[dirty["Flight#"].isin(truth["Flight#"])]

print res.shape
print len(res["Flight#"].unique())

sources = dirty["Source"].unique()

for s in sources:
    res_new = res[res["Source"] == s]
    print s + " - " + str(res_new.shape)


test = dirty["Flight#"].values
wanted = truth["Flight#"].values

res3 = test[np.logical_or.reduce([test[:] == x for x in wanted])]

#print res3.shape