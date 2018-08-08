from ml.datasets.Citations.Citation import Citation


data = Citation()

for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        if data.matrix_is_error[x,y]:
            print "column: " + str(y) + ": " + str(data.clean_pd.columns[y]) +  " clean: #" + str(data.clean_pd.values[x,y].encode('utf8')) + "# dirty: #" + str(data.dirty_pd.values[x,y].encode('utf8')) + "#"


#