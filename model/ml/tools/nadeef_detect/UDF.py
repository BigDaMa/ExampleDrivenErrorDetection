import time
from ml.configuration.Config import Config
import os
import numpy as np

class UDF:
    def __init__(self, column, java_violation_condition):
        self.type = 'udf'
        self.column = column.lower()

        self.name = str(long(time.time())) + str(np.random.randint(1000))

        #generate code

        file_content = """
        package qa.qcri.nadeef.test.udf;

        import qa.qcri.nadeef.core.datamodel.*;
        
        import java.util.ArrayList;
        import java.util.Collection;
        import java.util.List;
        
        public class GeneratedUDF""" + self.name + \
        """ extends SingleTupleRule {
        
            public boolean isNumeric(String str)  
            {  
              try  
              {  
                double d = Double.parseDouble(str);  
              }  
              catch(NumberFormatException nfe)  
              {  
                return false;  
              }  
              return true;  
            }
        
        
            /**
             * Detect rule with one tuple.
             *
             * @param tuple input tuple.
             * @return Violation set.
             */
            @Override
            public Collection<Violation> detect(Tuple tuple) {
                List<Violation> result = new ArrayList<>();
                String value = "";
                String col = """
        file_content += '"' + self.column + '";\n'
        file_content += 'value = (String)tuple.get(col);\n'
        #file_content += '}catch(NullPointerException e){}'
        file_content += 'if (' + java_violation_condition + """) {
                    Violation newViolation = new Violation(getRuleName());
                    newViolation.addCell(tuple.getCell(col));
                    result.add(newViolation);
                }
                return result;
            }

            /**
             * Repair of this rule.
             *
             *
             * @param violation violation input.
             * @return a candidate fix.
             */
            @Override
            public Collection<Fix> repair(Violation violation) {
                return null;
            }
        }
        """

        java_file = open(Config.get('nadeef.home') + '/test/src/qa/qcri/nadeef/test/udf/GeneratedUDF' + self.name + '.java', 'w+')
        java_file.write(file_content)
        java_file.close()

    def clean(self):
        os.remove(Config.get('nadeef.home') + '/test/src/qa/qcri/nadeef/test/udf/GeneratedUDF' + self.name + '.java')

    def __str__(self):
        rule_str = "qa.qcri.nadeef.test.udf.GeneratedUDF" + self.name

        return rule_str

    __repr__ = __str__
