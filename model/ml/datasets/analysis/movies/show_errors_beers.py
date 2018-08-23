from ml.datasets.BeersMohammad.Beers import Beers


data = Beers()

for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        if data.matrix_is_error[x,y]:
            #print str(data.dirty_pd.values[x,:])
            #if y == 4:
            print "column: " + str(y) + ": " + str(data.clean_pd.columns[y]) +  " clean: #" + str(data.clean_pd.values[x,y]) + "# dirty: #" + str(data.dirty_pd.values[x,y]) + "#"
            print "--------------------------------------"

