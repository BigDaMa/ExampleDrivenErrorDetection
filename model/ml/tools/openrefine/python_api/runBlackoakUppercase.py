from ml.tools.openrefine.python_api.RefineIT import RefineIT
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
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

        '''
        columns.append('State(String)')
        transformations.append('if(value.length() != 2, "error", value)')

        columns.append('ZIP(String)')
        transformations.append('if(value.length() != 5, "error", value)')

        '''
        columns.append('SSN(String)')
        transformations.append('if(isNumeric(value), value, "error")')

        '''
        columns.append('ZIP(String)')
        transformations.append('if(isNumeric(value), value, "error")')

        columns.append('City(String)')
        transformations.append('if(value == "SAN", "error", value)')

        columns.append('City(String)')
        transformations.append('if(value == "LOS", "error", value)')

        columns.append('City(String)')
        transformations.append('if(value == "SANTA", "error", value)')

        columns.append('City(String)')
        transformations.append('if(value == "EL", "error", value)')

        columns.append('City(String)')
        transformations.append('if(value == "NORTH", "error", value)')

        columns.append('City(String)')
        transformations.append('if(value == "PALM", "error", value)')

        columns.append('City(String)')
        transformations.append('if(value == "WEST", "error", value)')
        '''

        '''
        #more
        columns.append('FirstName(String)')
        transformations.append('if(contains(value, ","), "error", value)')
        columns.append('MiddleName(String)')
        transformations.append('if(contains(value, ","), "error", value)')
        columns.append('LastName(String)')
        transformations.append('if(contains(value, ","), "error", value)')
        columns.append('FirstName(String)')
        transformations.append('if(contains(value, "\'"), "error", value)')
        columns.append('MiddleName(String)')
        transformations.append('if(contains(value, "\'"), "error", value)')
        columns.append('LastName(String)')
        transformations.append('if(contains(value, "\'"), "error", value)')
        '''

        for i in range(len(columns)):
            print self.project.text_transform(column=columns[i], expression=transformations[i])

        response = self.project.export(export_format='tsv')

        myfile = open('/tmp/data_clean.tsv', 'wb')
        shutil.copyfileobj(response.fp, myfile)
        myfile.close()



if __name__ == '__main__':
    data = BlackOakDataSetUppercase()
    run = TutorialTestFacets(data)
