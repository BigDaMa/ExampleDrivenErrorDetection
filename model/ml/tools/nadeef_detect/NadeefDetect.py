import os
import random
import time
import csv

import psycopg2

from ml.configuration.Config import Config
from ml.tools.nadeef_detect.NadeefParse import NadeefParse


class NadeefDetect:

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
            header_name = header_name.replace(" ", "_").replace("-","_")
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
               "\t\t    \"type\" : \"" + rule.type + "\",\n" + \
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

        query = "CREATE TABLE temp_result_"+ name +" AS (Select tupleid,attribute, '\"' || value::text || '\"' From violation " \
                                                   "Where tablename = 'TB_" + name +"');"

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
        print "column_map: " + str(column_map)

        # read result from postgres and write to csv
        connection, cursor = self.connect_db(Config.get("nadeef.db"),
                                             Config.get("nadeef.db.user"),
                                             Config.get("nadeef.db.password"))

        ts = int(time.time())
        name = "dirty_data" + str(ts)
        csv_path = '/tmp/' + name + '.csv'
        #dirty.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL, quotechar='"')
        dirty.to_csv(csv_path, index=False, encoding='utf8')
        '''
        dirty.to_csv(csv_path,
               index=False,
               quoting=csv.QUOTE_ALL,
               escapechar='\\',
               quotechar="'",
               na_rep="")
        '''

        import os
        newline = os.linesep  # Defines the newline based on your OS.

        old_path1 = csv_path
        source_fp = open(old_path1, 'r')

        ts = int(time.time())
        name = "dirty_data_" + str(ts)
        csv_path = '/tmp/' + name + '.csv'

        target_fp = open(csv_path, 'w')
        first_row = True
        for row in source_fp:
            if first_row:
                row = row.replace("'", "").replace('"', '')
                first_row = False
            target_fp.write(row)
        target_fp.flush()
        source_fp.close()
        target_fp.close()

        #os.remove(old_path1)

        # compile
        os.system("cd " + Config.get("nadeef.home") + "/\n" + "ant all")

        for rule in rules:

            #create json configuration file
            conf = self.create_nadeef_conf(csv_path, rule)

            conf_path = '/tmp/nadeef_run_conf.json'
            with open(conf_path, 'w+') as f:
                f.write(conf)
            f.close()

            #run nadeef_detect
            strm = "printf \"\nload " + conf_path + "\n" \
                      "detect\n" \
                      "exit\n\" > /tmp/run_command.txt\n"
            os.system(strm)
            os.system(strm + "cd "+ Config.get("nadeef.home") +"/\n" \
                      "./nadeef.sh < /tmp/run_command.txt")

            result_path = "/tmp/nadeef_result.csv"
            self.get_result(connection, cursor, name,result_path)


            self.tool = NadeefParse(data, result_path, column_map)

            cur_fscore = self.tool.calculate_total_fscore()
            cur_precision = self.tool.calculate_total_precision()
            cur_recall = self.tool.calculate_total_recall()

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
            rule.clean()
            

        #final clean up
        self.clean_up_end(connection,name, cursor, csv_path)

        print "time: " + str(time_list)
        print "fscore: " + str(fscore)
        print "precision: " + str(precision)
        print "recall: " + str(recall)


