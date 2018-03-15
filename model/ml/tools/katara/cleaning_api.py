########################################
# Abstraction Layer
# Milad Abbaszadeh
# milad.abbaszadehjahromi@campus.tu-berlin.de
# October 2017
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
import os
import json
import re
import subprocess
import pandas
from ml.configuration.Config import Config
########################################


########################################
TOOLS_FOLDER = Config.get("abstractionlayer.tools")
########################################


########################################
def install_tools():
    """
    This method installs and configures the data cleaning tools.
    """
    for tool in os.listdir(TOOLS_FOLDER):
        if tool == "NADEEF":
            p = subprocess.Popen(["ant", "all"], cwd="{}/NADEEF".format(TOOLS_FOLDER), stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            p.communicate()
            postgress_username = Config.get("nadeef.db.user")
            postgress_password = Config.get("nadeef.db.password")
            nadeef_configuration_file = open("{}/NADEEF/nadeef.conf".format(TOOLS_FOLDER), "r")
            nadeef_configuration = nadeef_configuration_file.read()
            nadeef_configuration = re.sub("(database.username = )([\w\d]+)", "\g<1>{}".format(postgress_username),
                                          nadeef_configuration, flags=re.IGNORECASE)
            nadeef_configuration = re.sub("(database.password = )([\w\d]+)", "\g<1>{}".format(postgress_password),
                                          nadeef_configuration, flags=re.IGNORECASE)
            nadeef_configuration_file.close()
            nadeef_configuration_file = open("{}/NADEEF/nadeef.conf".format(TOOLS_FOLDER), "w")
            nadeef_configuration_file.write(nadeef_configuration)
        print "{} is installed.".format(tool)
########################################


########################################
def read_csv_dataset(dataset_path, header_exists=True):
    """
    The method reads a dataset from a csv file path.
    """
    if header_exists:
        dataset_dataframe = pandas.read_csv(dataset_path, sep=",", header="infer", encoding="utf-8", keep_default_na=False)
        return [dataset_dataframe.columns.get_values().tolist()] + dataset_dataframe.get_values().tolist()
    else:
        dataset_dataframe = pandas.read_csv(dataset_path, sep=",", header=None, encoding="utf-8", keep_default_na=False)
        return dataset_dataframe.get_values().tolist()


def write_csv_dataset(dataset_path, dataset_table):
    """
    The method writes a dataset to a csv file path.
    """
    dataset_dataframe = pandas.DataFrame(data=dataset_table[1:], columns=dataset_table[0])
    dataset_dataframe.to_csv(dataset_path, sep=",", header=True, index=False, encoding="utf-8")
########################################


########################################
def run_dboost(dataset_path, dboost_parameters):
    """
    This method runs dBoost on a dataset.
    """
    command = ["./{}/dBoost/dboost/dboost-stdin.py".format(TOOLS_FOLDER), "-F", ",", "--statistical", "0.5"]
    dboost_parameters[0] = "--" + dboost_parameters[0]
    command += dboost_parameters + [dataset_path]
    p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.communicate()
    return_list = []
    tool_results_path = "dboost_results.csv"
    if os.path.exists(tool_results_path):
        detected_cells_list = read_csv_dataset(tool_results_path, header_exists=False)
        cell_visited_flag = {}
        for row, column, value in detected_cells_list:
            i = int(row)
            j = int(column)
            v = value
            if (i, j) not in cell_visited_flag and i > 0:
                cell_visited_flag[(i, j)] = 1
                return_list.append([i, j, v])
        os.remove(tool_results_path)
    return return_list


def run_nadeef(dataset_path, nadeef_parameters):
    """
    This method runs NADEEF on a dataset.
    """
    dataset_table = read_csv_dataset(dataset_path)
    temp_dataset_path = os.path.abspath("nadeef_temp_dataset.csv")
    new_header = [a + " varchar(20000)" for a in dataset_table[0]]
    write_csv_dataset(temp_dataset_path, [new_header] + dataset_table[1:])
    column_index = {a: dataset_table[0].index(a) for a in dataset_table[0]}
    actual_nadeef_parameters = []
    for param in nadeef_parameters:
        actual_nadeef_parameters.append({"type": "fd", "value": [" | ".join(param)]})
    nadeef_clean_plan = {
        "source": {
            "type": "csv",
            "file": [temp_dataset_path]
        },
        "rule": actual_nadeef_parameters
    }
    nadeef_clean_plan_path = "nadeef_clean_plan.json"
    nadeef_clean_plan_file = open(nadeef_clean_plan_path, "w")
    json.dump(nadeef_clean_plan, nadeef_clean_plan_file)
    nadeef_clean_plan_file.close()
    p = subprocess.Popen(["./nadeef.sh"], cwd="{}/NADEEF".format(TOOLS_FOLDER), stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    process_output, process_errors = p.communicate("load ../../nadeef_clean_plan.json\ndetect\nexit\n")
    os.remove(nadeef_clean_plan_path)
    tool_results_path = re.findall("INFO: Export to (.*csv)", process_output)[0]
    return_list = []
    if os.path.exists(tool_results_path):
        detected_cells_list = read_csv_dataset(tool_results_path, header_exists=False)
        cell_visited_flag = {}
        for row in detected_cells_list:
            i = int(row[3])
            j = column_index[row[4]]
            v = row[5]
            if (i, j) not in cell_visited_flag:
                cell_visited_flag[(i, j)] = 1
                return_list.append([i, j, v])
        os.remove(tool_results_path)
    os.remove(temp_dataset_path)
    return return_list


def run_openrefine(dataset_path, openrefine_parameters):
    """
    This method runs OpenRefine on a dataset.
    """
    dataset_table = read_csv_dataset(dataset_path)
    columns_patterns_dictionary = {dataset_table[0].index(column): [] for column in dataset_table[0]}
    for column, pattern in openrefine_parameters:
        if column in dataset_table[0]:
            columns_patterns_dictionary[dataset_table[0].index(column)].append(pattern)
    return_list = []
    cell_visited_flag = {}
    for i, row in enumerate(dataset_table):
        if i == 0:
            continue
        for j, value in enumerate(row):
            for pattern in columns_patterns_dictionary[j]:
                if not re.findall(pattern, value, re.IGNORECASE | re.UNICODE):
                    if (i, j) not in cell_visited_flag:
                        cell_visited_flag[(i, j)] = 1
                        return_list.append([i, j, value])
    return return_list


def run_katara(dataset_path, katara_parameters):
    """
    This method runs KATARA on a dataset.
    """
    command = ["java", "-classpath",
               "{0}/KATARA/out/test/test:{0}/KATARA/KATARA/out/test/test/SimplifiedKATARA.jar".format(TOOLS_FOLDER),
               "simplied.katara.SimplifiedKATARAEntrance"]
    knowledge_base_path = os.path.abspath(katara_parameters[0])

    print(command)
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.communicate(dataset_path + "\n" + knowledge_base_path + "\n")
    return_list = []
    tool_results_path = "katara_output.csv"
    if os.path.exists(tool_results_path):
        detected_cells_list = read_csv_dataset(tool_results_path, header_exists=False)
        cell_visited_flag = {}
        for row, column in detected_cells_list:
            i = int(row)
            j = int(column)
            v = None
            if (i, j) not in cell_visited_flag and i > 0:
                cell_visited_flag[(i, j)] = 1
                return_list.append([i, j, v])
        os.remove(tool_results_path)
        os.remove("crowdclient-runtime.log")
    return return_list
########################################


########################################
def run_data_cleaning_job(data_cleaning_job):
    """
    This method runs the data cleaning job based on the input configuration.
    """
    dataset_path = ""
    if data_cleaning_job["dataset"]["type"] == "csv":
        dataset_path = os.path.abspath(data_cleaning_job["dataset"]["param"][0])

    return_list = []
    if data_cleaning_job["tool"]["name"] == "dboost":
        return_list = run_dboost(dataset_path, data_cleaning_job["tool"]["param"])
    if data_cleaning_job["tool"]["name"] == "nadeef":
        return_list = run_nadeef(dataset_path, data_cleaning_job["tool"]["param"])
    if data_cleaning_job["tool"]["name"] == "openrefine":
        return_list = run_openrefine(dataset_path, data_cleaning_job["tool"]["param"])
    if data_cleaning_job["tool"]["name"] == "katara":
        return_list = run_katara(dataset_path, data_cleaning_job["tool"]["param"])
    return return_list
########################################


########################################
