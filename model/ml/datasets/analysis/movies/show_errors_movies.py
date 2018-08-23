from ml.datasets.MoviesMohammad.Movies import Movies


data = Movies()

for x in range(data.shape[0]):

    for y in range(data.shape[1]):
        if data.matrix_is_error[x,y]:
            #print str(data.clean_pd.values[x, :])
            print "row" + str(x) + " column: " + str(y) + ": " + str(data.clean_pd.columns[y]) +  " clean: #" + str(data.clean_pd.values[x,y]) + "# dirty: #" + str(data.dirty_pd.values[x,y]) + "#"


# len(Year) =4
# duration: not "hr"
# Genre: not "&"
# len(RatingValue) = 3
# len(Id) = 9