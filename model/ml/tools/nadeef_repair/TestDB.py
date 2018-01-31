import psycopg2


def connect_db(dbname, user, password):
    conn_string = "host='localhost' dbname='" + dbname + "' user='" + user + "' password='" + password + "'"
    # print the connection string we will use to connect
    print "Connecting to database\n	->%s" % (conn_string)

    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)

    # conn.cursor will return a cursor object, you can use this cursor to perform queries
    cursor = conn.cursor()

    return conn, cursor

connection, cursor = connect_db('nadeef_repair','felix','felix')

name = "dirty_DATA1500854412"

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
#cursor.close()
connection.commit()

result_path = "/tmp/nadeef_result.csv"

result_file = open(result_path, 'w')

cursor.copy_to(result_file, "temp_result_" + name, sep=",")

result_file.close()

from ml.tools.nadeef_repair.NadeefMe import NadeefMe
from ml.datasets.blackOak import BlackOakDataSet

tool = NadeefMe(BlackOakDataSet(), result_path)

print "Fscore: " + str(tool.calculate_total_fscore())
print "Precision: " + str(tool.calculate_total_precision())
print "Recall: " + str(tool.calculate_total_recall())