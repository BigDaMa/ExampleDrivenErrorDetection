from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np


def folder_to_frame(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]

    list_frames = []

    for f in files:
        data = pd.read_csv(path + f, sep='\t', header=None)
        list_frames.append(data)

    result = pd.concat(list_frames)
    return result

mypath = "/home/felix/SequentialPatternErrorDetection/flight/data/clean_flight/"
dirty = folder_to_frame(mypath)

mypath = "/home/felix/SequentialPatternErrorDetection/flight/data/flight_truth/"
ground_truth = folder_to_frame(mypath)

print dirty.shape
print ground_truth.shape