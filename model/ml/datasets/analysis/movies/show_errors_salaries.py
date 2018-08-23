#from ml.datasets.salary_data.Salary import Salary
from ml.datasets.LarysaSalaries.Salaries import Salaries


data = Salaries()

for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        if data.matrix_is_error[x,y]:
            #print str(data.dirty_pd.values[x,:])
            print "column: " + str(y) + ": " + str(data.clean_pd.columns[y]) +  " clean: #" + str(data.clean_pd.values[x,y]) + "# dirty: #" + str(data.dirty_pd.values[x,y]) + "#"
            print "--------------------------------------"

