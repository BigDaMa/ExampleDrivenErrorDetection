from ml.datasets.MoviesMohammad.Movies import Movies
import numpy as np

data = Movies()

print list(np.sum(data.matrix_is_error, axis=0) / float(data.shape[0]))
print data.clean_pd.columns


#name = data.dirty_pd['Name'].values
name = data.dirty_pd['Id'].values
dir = data.clean_pd['Director'].values

my_file = open('/tmp/director.rel.txt', 'w+')
for i in range(data.shape[0]):
    my_file.write(name[i] +"\thasdirector\t" + dir[i] + "\n")
my_file.close()


dir = data.clean_pd['Duration'].values



my_file = open('/tmp/duration.rel.txt', 'w+')
for i in range(data.shape[0]):
    my_value = dir[i]
    if len(my_value) == 0:
        my_value = '#NULL#'
    my_file.write(name[i] +"\thasduration\t" + my_value + "\n")
my_file.close()