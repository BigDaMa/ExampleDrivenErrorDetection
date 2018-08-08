class FD:
    def __init__(self, determinant_set, dependant, data=None):
        self.determinant_set = determinant_set
        self.dependant = dependant
        self.data = data
        self.type = 'fd'

    def __str__(self):
        rule_str = ""

        if self.data == None:
            for determinant in self.determinant_set:
                rule_str += determinant + ","

            rule_str = rule_str[:-1]

            rule_str += " | "

            rule_str += self.dependant

        return rule_str

    def clean(self):
        pass

    __repr__ = __str__
