import numpy as np
import pandas as pd

from ml.tools.Tool import Tool


class NadeefMe(Tool):

    '''
    Start postgres:
    sudo -u postgres psql postgres

    \connect nadeef_repair


    RUN SQL:

    COPY
        (Select tbl.* From audit tbl
        Inner Join
        (
          Select tupleid, attribute, tablename, MIN(time) MinPoint From audit Group By tupleid, attribute, tablename
        )tbl1
        On tbl1.tupleid=tbl.tupleid and tbl1.attribute=tbl.attribute and tbl1.tablename=tbl.tablename
        Where tbl.tablename = 'TB_BLACKOAK1' and tbl1.MinPoint=tbl.time)
    TO '/tmp/blackoak_nadeef_new.csv' DELIMITER ',' CSV HEADER;

    and change tablename


    '''


    def __init__(self, data, path, column_map):
        path_to_tool_result = path

        outliers = pd.read_csv(path_to_tool_result, header=None, na_filter=False, dtype={5:object, 6:object})

        print outliers.columns

        def run(x):
            if "'" in str(x):
                return x[1:len(x)-1]
            else:
                return x


        print outliers


        outliers[outliers.columns[5]] = outliers[outliers.columns[5]].apply(lambda x : run(x))

        matrix_outliers = outliers.values

        matrix_detected = np.zeros(data.shape, dtype=bool)

        for i in range(len(matrix_outliers)):
                tuple_id = matrix_outliers[i][2]
                row_id = tuple_id - 1
                attribute_id = column_map[matrix_outliers[i][4]]
                old_value = matrix_outliers[i][5]

                assert old_value == data.dirty_pd.values[row_id][attribute_id], " #" + str(old_value) + "# vs: #" + str(data.dirty_pd.values[row_id][attribute_id]) + "#"
                matrix_detected[row_id][attribute_id] = True
        super(NadeefMe, self).__init__("Nadeef_me", data, matrix_detected)


    def validate(self):
        print "test"

    def fillna_df(self, df):
        if df[df.columns[5]].dtypes.name == "object":
            df[df.columns[5]] = df[df.columns[5]].fillna('')
        else:
            raise Exception('not implemented')
        #todo if numeric
        return df


