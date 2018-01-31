import os
import random
import time

from ml.datasets.DataSetBasic import DataSetBasic
from ml.datasets.blackOak import BlackOakDataSet
from ml.tools.dboost.DBoostMe import DBoostMe


def sample(x, n):
    random_index = random.sample(x.index, n)
    return x.ix[random_index], random_index

data = BlackOakDataSet()

n = 100 #data.shape[0]

data_sample, random_index = sample(data.dirty_pd, n)

data_sample_ground_truth = data.matrix_is_error[random_index]

sample_file = "/tmp/data_sample.csv"
result_file = "/tmp/dboostres.csv"

data_sample.to_csv(sample_file, index=False)


total_start_time = time.time()

nn = 10
for step in range(nn):
        gaus = ((2.0 - 0.0) / nn) * step
        stat = 0.5

        command = "python3 /home/felix/dBoost/dboost/dboost-stdin.py -F ','  --gaussian " + str(gaus) + " --statistical " + str(stat) + " " + sample_file + " > " + result_file

        os.system(command)


        our_sample_data = DataSetBasic(data.name + " random"+ str(n), data_sample, data_sample_ground_truth)

        run = DBoostMe(our_sample_data, result_file)

        print "--gaussian " + str(gaus) + " --statistical "  + str(stat) + " -> Fscore: " + str(run.calculate_total_fscore())
        print "Precision: " + str(run.calculate_total_precision())
        print "Recall: " + str(run.calculate_total_recall())

runtime = (time.time() - total_start_time)

print runtime