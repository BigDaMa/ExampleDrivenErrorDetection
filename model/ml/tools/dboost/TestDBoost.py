import os
import random
import time

import numpy as np

from ml.configuration.Config import Config
from ml.datasets.DataSetBasic import DataSetBasic
from ml.tools.dboost.DBoostMe import DBoostMe


def run_gaussian_stat(gaussian, statistical, sample_file = "/tmp/data_sample.csv", result_file = "/tmp/dboostres.csv"):
    command = "python3 " + Config.get("dboost.py") + " -F ','  --gaussian " + str(
        gaussian) + " --statistical " + str(statistical) + " '" + sample_file + "' > '" + result_file + "'"
    os.system(command)


def run_histogram_stat(peak, outlier, statistical, sample_file = "/tmp/data_sample.csv", result_file = "/tmp/dboostres.csv"):
    command = "python3 " + Config.get("dboost.py") + " -F ','  --histogram " + str(
        peak) + " " + str(outlier) + " --statistical " + str(statistical) + " '" + sample_file + "' > '" + result_file + "'"

    os.system(command)

def run_mixture_stat(n_subpops, threshold, statistical, sample_file ="/tmp/data_sample.csv", result_file ="/tmp/dboostres.csv"):
    command = "python3 -W ignore " + Config.get("dboost.py") + " -F ','  --mixture " + str(
        n_subpops) + " " + str(threshold) + " --statistical " + str(statistical) + " '" + sample_file + "' > '" + result_file + "'"

    os.system(command)


def sample(x, n):
    random_index = random.sample(x.index, n)
    return x.ix[random_index], random_index


def search_gaussian_stat(data, data_sample, data_sample_ground_truth,sample_file, result_file, gaussian_range, statistical_range, write_out=False):
    best_params = {}
    best_fscore = 0.0
    precision = 0.0
    recall = 0.0

    for g in gaussian_range:
        for s in statistical_range:
            run_gaussian_stat(g, s, sample_file, result_file)

            our_sample_data = DataSetBasic(data.name + " random" + str(data_sample.shape[0]), data_sample, data_sample_ground_truth)

            run = DBoostMe(our_sample_data, result_file)

            current_fscore = run.calculate_total_fscore()
            current_precision = run.calculate_total_precision()
            current_recall = run.calculate_total_recall()

            if write_out:
                run.write_detected_matrix(Config.get("logging.folder") + "/out/dboost" + '/dboost_gausian_' + data.name + '_gausian' + str(g) + '_stat_' + str(s) + '.npy')

            print "--gaussian " + str(g) + " --statistical " + str(s)
            print "Fscore: " + str(current_fscore)
            print "Precision: " + str(run.calculate_total_precision())
            print "Recall: " + str(run.calculate_total_recall())

            if current_fscore >= best_fscore:
                best_fscore = current_fscore
                precision = current_precision
                recall = current_recall
                best_params['gaussian'] = g
                best_params['statistical'] = s

    return best_params, best_fscore, precision, recall


def search_histogram_stat(data, data_sample, data_sample_ground_truth,sample_file, result_file, peak_s, outlier_s, statistical_range, write_out=False):
    best_params = {}
    best_fscore = 0.0
    precision = 0.0
    recall = 0.0

    for p in peak_s:
        for o in outlier_s:
            for s in statistical_range:
                run_histogram_stat(p, o, s, sample_file, result_file)

                our_sample_data = DataSetBasic(data.name + " random" + str(data_sample.shape[0]), data_sample, data_sample_ground_truth)

                run = DBoostMe(our_sample_data, result_file)

                current_fscore = run.calculate_total_fscore()
                current_precision = run.calculate_total_precision()
                current_recall = run.calculate_total_recall()

                if write_out:
                    run.write_detected_matrix(
                        Config.get("logging.folder") + "/out/dboost" + '/dboost_histogram_' + data.name + '_peak' + str(
                            p) + '_outlier_' + str(o) + '_stat_' + str(s) + '.npy')

                print "peak: " + str(p) + " outlier: " + str(o) + " --statistical " + str(s)
                print "Fscore: " + str(current_fscore)
                print "Precision: " + str(run.calculate_total_precision())
                print "Recall: " + str(run.calculate_total_recall())

                if current_fscore >= best_fscore:
                    best_fscore = current_fscore
                    precision = current_precision
                    recall = current_recall
                    best_params['peak'] = p
                    best_params['outlier'] = o
                    best_params['statistical'] = s

    return best_params, best_fscore, precision, recall


def search_mixture_stat(data,
                        data_sample,
                        data_sample_ground_truth,
                        sample_file,
                        result_file,
                        n_subpops_s,
                        threshold_s,
                        statistical_range,
                        write_out=False):
    best_params = {}
    best_fscore = 0.0
    precision = 0.0
    recall = 0.0

    for p in n_subpops_s:
        for t in threshold_s:
            for s in statistical_range:
                run_mixture_stat(p, t, s, sample_file, result_file)

                our_sample_data = DataSetBasic(data.name + " random" + str(data_sample.shape[0]), data_sample, data_sample_ground_truth)

                run = DBoostMe(our_sample_data, result_file)

                current_fscore = run.calculate_total_fscore()
                current_precision = run.calculate_total_precision()
                current_recall = run.calculate_total_recall()

                if write_out:
                    run.write_detected_matrix(
                        Config.get("logging.folder") + "/out/dboost" + '/dboost_' + data.name + '_mixture_subpop' + str(
                            p) + '_threshold_' + str(t) + '_stat_' + str(s) + '.npy')

                print "n_subpops: " + str(p) + " threshold: " + str(t) + " --statistical " + str(s)
                print "Fscore: " + str(current_fscore)
                print "Precision: " + str(run.calculate_total_precision())
                print "Recall: " + str(run.calculate_total_recall())

                if current_fscore >= best_fscore:
                    best_fscore = current_fscore
                    precision = current_precision
                    recall = current_recall
                    best_params['n_subpops'] = p
                    best_params['threshold'] = t
                    best_params['statistical'] = s

    return best_params, best_fscore, precision, recall


def get_files(data_sample):
    sample_file = "/tmp/data_sample_" + str(time.time()) + "_" + str(random.randint(0,1000)) + ".csv"
    data_sample.to_csv(sample_file, index=False, encoding='utf8')
    result_file = "/tmp/dboostres_" + str(time.time()) + "_" + str(random.randint(0,1000)) + ".csv"

    return sample_file, result_file

def grid_search_by_sample_gaussian(data, sample_size, steps):

    n = sample_size

    data_sample, random_index = sample(data.dirty_pd, n)

    data_sample_ground_truth = data.matrix_is_error[random_index, :]

    sample_file, result_file = get_files(data_sample)

    total_start_time = time.time()

    gaussian_range = [((4.0 - 0.0) / steps) * step for step in range(steps)]
    statistical_range = [0.5]

    best_params, best_fscore_1, precision_1, recall_1 = search_gaussian_stat(data, data_sample, data_sample_ground_truth, sample_file, result_file, gaussian_range, statistical_range)

    os.remove(sample_file)

    runtime = (time.time() - total_start_time)

    print "grid search runtime: " + str(runtime)

    return best_params


def grid_search_by_sample_hist(data, sample_size, steps):

    n = sample_size

    data_sample, random_index = sample(data.dirty_pd, n)

    data_sample_ground_truth = data.matrix_is_error[random_index, :]

    sample_file, result_file = get_files(data_sample)


    total_start_time = time.time()

    peak_range = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    outlier_range = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    statistical_range = [0.5]

    best_params, best_fscore_1, precision_1, recall_1 = search_histogram_stat(data, data_sample, data_sample_ground_truth, sample_file, result_file, peak_range, outlier_range, statistical_range)

    os.remove(sample_file)

    runtime = (time.time() - total_start_time)

    print "grid search runtime: " + str(runtime)

    return best_params


def grid_search_by_sample_mixture(data, sample_size, steps):

    n = sample_size

    data_sample, random_index = sample(data.dirty_pd, n)

    data_sample_ground_truth = data.matrix_is_error[random_index, :]

    sample_file, result_file = get_files(data_sample)


    total_start_time = time.time()

    n_subpops_range = [1,2,3]
    threshold_range = [0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    statistical_range = [0.5]

    best_params, best_fscore_1, precision_1, recall_1 = search_mixture_stat(data, data_sample, data_sample_ground_truth, sample_file, result_file, n_subpops_range, threshold_range, statistical_range)


    runtime = (time.time() - total_start_time)

    print "grid search runtime: " + str(runtime)

    os.remove(sample_file)

    return best_params


def run_params_gaussian(data, params):
    #n = data.shape[0]

    #data_sample, random_index = sample(data.dirty_pd, n)
    data_sample = data.dirty_pd

    #data_sample_ground_truth = data.matrix_is_error[random_index, :]
    data_sample_ground_truth = data.matrix_is_error

    sample_file, result_file = get_files(data_sample)

    total_start_time = time.time()

    gaussian_range = [params['gaussian']]
    statistical_range = [params['statistical']]

    print "Run on all: "
    _, best_fscore, precision, recall = search_gaussian_stat(data, data_sample, data_sample_ground_truth, sample_file, result_file, gaussian_range,
                                       statistical_range, True)

    runtime = (time.time() - total_start_time)

    print "runtime for one run on all data: " + str(runtime)

    os.remove(sample_file)

    return best_fscore, precision, recall

def run_params_hist(data, params):
    # n = data.shape[0]

    # data_sample, random_index = sample(data.dirty_pd, n)
    data_sample = data.dirty_pd

    # data_sample_ground_truth = data.matrix_is_error[random_index, :]
    data_sample_ground_truth = data.matrix_is_error

    sample_file, result_file = get_files(data_sample)

    total_start_time = time.time()

    peak_range = [params['peak']]
    outlier_range = [params['outlier']]
    statistical_range = [params['statistical']]

    print "Run on all: "
    _, best_fscore, precision, recall = search_histogram_stat(data, data_sample, data_sample_ground_truth, sample_file, result_file, peak_range, outlier_range,
                                       statistical_range, True)

    runtime = (time.time() - total_start_time)

    print "runtime for one run on all data: " + str(runtime)

    os.remove(sample_file)

    return best_fscore, precision, recall


def run_params_mixture(data, params):
    # n = data.shape[0]

    # data_sample, random_index = sample(data.dirty_pd, n)
    data_sample = data.dirty_pd

    # data_sample_ground_truth = data.matrix_is_error[random_index, :]
    data_sample_ground_truth = data.matrix_is_error

    sample_file, result_file = get_files(data_sample)

    total_start_time = time.time()

    n_subpops_range = [params['n_subpops']]
    threshold_range = [params['threshold']]
    statistical_range = [params['statistical']]

    print "Run on all: "
    _, best_fscore, precision, recall = search_mixture_stat(data, data_sample, data_sample_ground_truth, sample_file, result_file, n_subpops_range, threshold_range,
                                       statistical_range, True)

    runtime = (time.time() - total_start_time)

    print "runtime for one run on all data: " + str(runtime)

    os.remove(sample_file)

    return best_fscore, precision, recall


def test_gaussian(data, sample_size, steps):
    best_params = grid_search_by_sample_gaussian(data, sample_size, steps)
    best_fscore_all, precision, recall = run_params_gaussian(data, best_params)
    return best_fscore_all, precision, recall, best_params

def test_hist(data, sample_size, steps):
    best_params = grid_search_by_sample_hist(data, sample_size, steps)
    best_fscore_all, precision, recall = run_params_hist(data, best_params)
    return best_fscore_all, precision, recall, best_params

def test_mixture(data, sample_size, steps):
    best_params = grid_search_by_sample_mixture(data, sample_size, steps)
    best_fscore_all, precision, recall = run_params_mixture(data, best_params)
    return best_fscore_all, precision, recall, best_params

def test_multiple_sizes(data, steps, N=3, sizes = [10, 100, 1000], run_algo_function = test_gaussian, log_file=None):
    avg_times = []
    avg_fscores = []
    avg_precision = []
    avg_recall = []

    std_fscores = []
    std_precision = []
    std_recall = []

    for t in sizes:
        times = []
        fscore = []
        prec_list = []
        rec_list = []

        with open(log_file, "a") as myfile:
            myfile.write("training size: " + str(t) + "\n\n")

        for i in range(N):
            total_start_time = time.time()

            best_fscore_all, precision, recall, best_params = run_algo_function(data=data, sample_size=t, steps=steps)
            runtime = (time.time() - total_start_time)

            times.append(runtime)
            fscore.append(best_fscore_all)
            prec_list.append(precision)
            rec_list.append(recall)

            if log_file != None:
                with open(log_file, "a") as myfile:
                    myfile.write(str(best_params) + ", " +
                                 str(runtime) + ", " +
                                 str(precision) + ", " +
                                 str(recall) + ", " +
                                 str(best_fscore_all) + "\n")


        avg_times.append(np.mean(times))
        avg_fscores.append(np.mean(fscore))
        avg_precision.append(np.mean(prec_list))
        avg_recall.append(np.mean(rec_list))

        std_fscores.append(np.std(fscore))
        std_precision.append(np.std(prec_list))
        std_recall.append(np.std(rec_list))

    print "labelled rows: " + str(sizes)
    print "time: " + str(avg_times)
    print "avg fscore: " + str(avg_fscores)
    print "avg precision: " + str(avg_precision)
    print "avg recall: " + str(avg_recall)

    print "std fscore: " + str(std_fscores)
    print "std precision: " + str(std_precision)
    print "std recall: " + str(std_recall)

    if log_file != None:
        with open(log_file, "a") as myfile:
            myfile.write("labelled rows: " + str(sizes) + '\n' + "time: " + str(avg_times)+ '\n' + "avg fscore: " +
                         str(avg_fscores)+ '\n' + "avg precision: " + str(avg_precision)+ '\n' + "avg recall: " +
                         str(avg_recall)+ '\n' + "std fscore: " + str(std_fscores)+ '\n' + "std precision: " +
                         str(std_precision)+ '\n' + "std recall: " + str(std_recall)+ '\n')

    return avg_times, avg_fscores, avg_precision, avg_recall, std_fscores, std_precision, std_recall



def toLatex(defined_range_labeled_cells, avg_times, avg_fscores, avg_precision, avg_recall, std_fscores, std_precision, std_recall, log_file):
    #chart
    latex_str = "\n\\addplot+[mark=x] coordinates{"
    for i in range(len(defined_range_labeled_cells)):
        latex_str += '(' + str(defined_range_labeled_cells[i]) + ',' + str(avg_fscores[i]) +')'
    latex_str += '};\n\n'

    #table
    table_str = ''
    for i in range(len(defined_range_labeled_cells)):
        table_str += 'labled cells: ' + str(defined_range_labeled_cells[i]) + '\n\n'
        table_str += str(avg_precision[i]) + ' $\pm$ ' + str(std_precision[i]) +  ' & ' \
                     + str(avg_recall[i]) + ' $\pm$ ' + str(std_recall[i]) + ' & ' \
                     + str(avg_fscores[i]) + ' $\pm$ ' + str(std_fscores[i])  \
                     + '&&' + '\n\n'

    with open(log_file, "a") as myfile:
        myfile.write(latex_str + table_str)






def test_multiple_sizes_gaussian(data, steps, N=3, sizes = [10, 100, 1000], log_file=None):
    return test_multiple_sizes(data, steps, N, sizes, test_gaussian, log_file)

def test_multiple_sizes_hist(data, steps, N=3, sizes = [10, 100, 1000], log_file=None):
    return test_multiple_sizes(data, steps, N, sizes, test_hist, log_file)

def test_multiple_sizes_mixture(data, steps, N=3, sizes = [10, 100, 1000], log_file=None):
    return test_multiple_sizes(data, steps, N, sizes, test_mixture, log_file)

