from ml.tools.openrefine.python_api.RefineIT import RefineIT
from ml.datasets.RestaurantMohammad.Restaurant import Restaurant
from ml.tools.openrefine.OpenRefine import OpenRefine
import shutil
import time

class TutorialTestFacets(RefineIT):
    project_options = {}

    def __init__(self, data):
        self.data = data
        self.project_file = '/tmp/data_dirty.csv'

        self.data.dirty_pd.to_csv(self.project_file, index=False, header=True)

        start_time = time.time()

        self.setUp()
        self.run_transforms()
        self.tearDown()

        runtime = (time.time() - start_time)

        tool = OpenRefine('/tmp/data_clean.tsv', data=data)

        print "Runtime: " + str(runtime)
        print "Fscore: " + str(tool.calculate_total_fscore())
        print "Precision: " + str(tool.calculate_total_precision())
        print "Recall: " + str(tool.calculate_total_recall())

    def run_transforms(self):
        print self.data.dirty_pd.columns

        columns = []
        transformations = []

        columns.append('website')
        transformations.append('if(not(or(startsWith(toString(value), "http"), value == null)), "error", value)')


        for i in range(len(columns)):
            print self.project.text_transform(column=columns[i], expression=transformations[i])

        response = self.project.export(export_format='tsv')

        myfile = open('/tmp/data_clean.tsv', 'wb')
        shutil.copyfileobj(response.fp, myfile)
        myfile.close()



if __name__ == '__main__':
    data = Restaurant()
    run = TutorialTestFacets(data)
