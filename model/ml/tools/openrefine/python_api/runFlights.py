from ml.tools.openrefine.python_api.refinetest import RefineIT
from ml.datasets.flights.FlightHoloClean import FlightHoloClean
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
        for columni in range(2, data.shape[1]):
            print self.project.text_transform(column=data.dirty_pd.columns[columni], expression='if(isNotNull(value.match(/\d+\:\d+ [p,a]\.m\./)), value, "error")')

        '''
        for columni in range(2, data.shape[1]):
            print self.project.text_transform(column=data.dirty_pd.columns[columni], expression='if(isNull(value), "error", value)')
        '''

        response = self.project.export(export_format='tsv')

        myfile = open('/tmp/data_clean.tsv', 'wb')
        shutil.copyfileobj(response.fp, myfile)
        myfile.close()



if __name__ == '__main__':
    data = FlightHoloClean()
    run = TutorialTestFacets(data)
