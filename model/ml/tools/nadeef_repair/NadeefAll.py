import os
import random
import time
from sets import Set

import psycopg2

from ml.datasets.blackOak import BlackOakDataSet
from ml.tools.nadeef_repair.FD import FD
from ml.tools.nadeef_repair.NadeefMe import NadeefMe


class NadeefAll:

    def correct_header(self, df):
        new_header = {}
        column_map = {}

        i = 0
        for h in df.columns:
            pos = h.find('(')
            if pos >= 0:
                header_name = h[0:pos]
            else:
                header_name = h
            header_name = header_name.replace(" ", "_")
            new_header[h] = header_name + " string"
            column_map[header_name.lower()] = i
            i += 1

        new_df = df.rename(index=str, columns=new_header)

        return new_df, column_map

    def sample(self, x, n):
        random_index = random.sample(x.index, n)
        return x.ix[random_index], random_index


    def create_nadeef_conf(self, csv_path, rule):
        conf = "{\n" + \
               "    \"source\" : {\n" + \
               "        \"type\" : \"csv\",\n" + \
               "        \"file\" : [\""+ csv_path +"\"]\n" + \
               "    },\n" + \
               "    \"rule\" : [\n" + \
               "        {\n" + \
               "\t\t    \"type\" : \"fd\",\n" + \
               "            \"value\" : [\"" + str(rule) + "\"]\n" + \
               "        }\n" + \
               "    ]\n" + \
               "}"

        return conf


    def connect_db(self, dbname, user, password):
        conn_string = "host='localhost' dbname='"+ dbname +"' user='" + user + "' password='" + password + "'"
        # print the connection string we will use to connect
        print "Connecting to database\n	->%s" % (conn_string)

        # get a connection, if a connect cannot be made an exception will be raised here
        conn = psycopg2.connect(conn_string)

        # conn.cursor will return a cursor object, you can use this cursor to perform queries
        cursor = conn.cursor()

        return conn, cursor

    def clean_up(self, connection, name, cursor, result_path):
        cursor.execute("DROP TABLE temp_result_"+ name + ";")

        connection.commit()

        # remove result file:
        os.remove(result_path)

    def clean_up_end(self, connection, name, cursor, csv_path):
        cursor.execute("DROP TABLE tb_" + name + ";")

        cursor.execute("TRUNCATE violation;")
        cursor.execute("TRUNCATE repair;")
        cursor.execute("TRUNCATE audit;")

        connection.commit()

        # remove result file:
        os.remove(csv_path)

    def get_result(self, connection, cursor, name, result_path):

        name = name.upper()

        query = "CREATE TABLE temp_result_"+ name +" AS (Select tbl.* From audit tbl\n" + \
			"Inner Join\n" + \
			"(\n" + \
			"  Select tupleid,attribute,tablename,MIN(time) MinPoint From audit Group By tupleid,attribute," + \
			"tablename\n" + \
			")tbl1\n" + \
			"On tbl1.tupleid=tbl.tupleid and tbl1.attribute=tbl.attribute and tbl1.tablename=tbl.tablename\n" + \
			"Where tbl.tablename = 'TB_" + name +"' and tbl1.MinPoint=tbl.time);"

        cursor.execute(query)
        connection.commit()

        result_file = open(result_path, 'w')

        cursor.copy_to(result_file, "temp_result_" + name, sep=",")


        result_file.close()




    def __init__(self, data, rules, log_file=None):

        time_list = []
        fscore = []
        precision = []
        recall = []

        total_start_time = time.time()

        dirty = data.dirty_pd

        #remove () from header
        # add ' string' to each header
        dirty, column_map = self.correct_header(dirty)

        print dirty.columns
        print column_map

        # read result from postgres and write to csv
        connection, cursor = self.connect_db('nadeef', 'felix', 'felix')

        ts = int(time.time())
        name = "dirty_data" + str(ts)

        csv_path = '/tmp/' + name + '.csv'
        dirty.to_csv(csv_path, index=False)

        for rule in rules:

            #create json configuration file
            conf = self.create_nadeef_conf(csv_path, rule)

            conf_path = '/tmp/nadeef_run_conf.json'
            with open(conf_path, 'w+') as f:
                f.write(conf)
            f.close()

            #run nadeef_repair
            strm = "printf \"\nload " + conf_path + "\n" \
                      "run 0\n" \
                      "exit\n\" > /tmp/run_command.txt\n"
            os.system(strm)
            os.system(strm + "cd /home/felix/NADEEF/\n" \
                      "./nadeef.sh < /tmp/run_command.txt")

            result_path = "/tmp/nadeef_result.csv"
            self.get_result(connection, cursor, name,result_path)

            tool = NadeefMe(data, result_path, column_map)

            cur_fscore = tool.calculate_total_fscore()
            cur_precision = tool.calculate_total_precision()
            cur_recall = tool.calculate_total_recall()

            fscore.append(cur_fscore)
            precision.append(cur_precision)
            recall.append(cur_recall)


            print "Fscore: " + str(cur_fscore)
            print "Precision: " + str(cur_precision)
            print "Recall: " + str(cur_recall)

            runtime = (time.time() - total_start_time)
            time_list.append(runtime)

            print "runtime for rule: " + str(rule) + " => " + str(runtime)

            if log_file != None:
                with open(log_file, "a") as myfile:
                    myfile.write(str(rule) + ", " +
                                 str(runtime) + ", " +
                                 str(cur_precision) + ", " +
                                 str(cur_recall) + ", " +
                                 str(cur_fscore) + "\n")

            #clean up
            self.clean_up(connection, name, cursor, result_path)

        #final clean up
        self.clean_up_end(connection,name, cursor, csv_path)

        print "time: " + str(time_list)
        print "fscore: " + str(fscore)
        print "precision: " + str(precision)
        print "recall: " + str(recall)




if __name__ == '__main__':

    data = BlackOakDataSet()

    rules = []
    rules.append(FD(Set(["ZIP"]), "City"))
    rules.append(FD(Set(["ZIP"]), "State"))

    nadeef = NadeefAll(BlackOakDataSet(),rules)

    #data = HospitalHoloClean()
    #nadeef_repair = NadeefAll(BlackOakDataSet())
