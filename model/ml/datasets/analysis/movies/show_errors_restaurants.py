from ml.datasets.RestaurantMohammad.Restaurant import Restaurant


data = Restaurant()

for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        if data.matrix_is_error[x,y]:
            print "column: " + str(y) + ": " + str(data.clean_pd.columns[y]) +  " clean: #" + str(data.clean_pd.values[x,y]) + "# dirty: #" + str(data.dirty_pd.values[x,y]) + "#"


# if(not(or(startsWith(toString(value), 'http'), value == null)), "error", value)