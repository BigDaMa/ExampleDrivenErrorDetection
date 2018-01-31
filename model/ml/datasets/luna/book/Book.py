import pandas as pd
import numpy as np

from ml.datasets.DataSet import DataSet


class Book(DataSet):
    name = "Book"
    def __init__(self):
        path_to_mohammad_git_repo = "/home/felix/SequentialPatternErrorDetection/luna/book/"

        path_dirty = path_to_mohammad_git_repo + "book.txt"
        path_clean = path_to_mohammad_git_repo + "book_silver.txt"

        dirty_pd = pd.read_csv(path_dirty, sep='\t', header=None, dtype=object, na_filter=False, names=['Source','ISBN','Title','Author_list'])
        clean_authors = pd.read_csv(path_clean, sep='\t', header=None, dtype=object, na_filter=False)

        #print dirty_pd

        dirty_pd = dirty_pd.sort_values(['ISBN', 'Source'], ascending=[1, 1])


        clean_pd = dirty_pd.copy()



        for t in range(clean_authors.shape[0]):
            id = clean_authors.values[t,0]
            mask = clean_pd[clean_pd.columns[1]] == id
            clean_pd.loc[mask, clean_pd.columns[3]] = clean_authors.values[t,1]

        super(Book, self).__init__(Book.name, dirty_pd, clean_pd)

    def validate(self):
        print "validate"

if __name__ == '__main__':

    data = Book()

    print np.sum(data.matrix_is_error)