from sets import Set

import pandas as pd

from ml.datasets.DataSet import DataSet


class IQContest(DataSet):
    def __init__(self):
        path_to_dirty = "/home/felix/SequentialPatternErrorDetection/IQContest/IQContest/data/COLUMBA.TXT"
        path_to_clean = "/home/felix/SequentialPatternErrorDetection/IQContest/IQContest/data/OPENMMS.TXT"

        attribute_names = ['PDB-ID',
                           'NAME',
                           'DEPOSITION_YEAR',
                           'DEPOSITION_DATE',
                           'RELEASE_DATE',
                           'AUTHORS',
                           'STRUCTURE_METHOD',
                           'RESOLUTION',
                           'REFINEMENT_PROGRAM',
                           'STRUCTURES',
                           'CHAINS',
                           'CHAIN_RESIDUES',
                           'CHAIN_ATOMS',
                           'HETERO_GROUPS',
                           'HETERO_ATOMS']

        dirty_pd = pd.read_csv(path_to_dirty, sep='\t', dtype=object, header=None, names = attribute_names)
        clean_pd = pd.read_csv(path_to_clean, sep='\t', dtype=object, header=None, names = attribute_names)

        print dirty_pd.shape
        print clean_pd.shape

        a = Set(dirty_pd[dirty_pd.columns[0]].unique())
        b = Set(clean_pd[clean_pd.columns[0]].unique())
        c = list(a.intersection(b))

        dirty_pd = dirty_pd[dirty_pd[dirty_pd.columns[0]].isin(c)]
        clean_pd = clean_pd[clean_pd[clean_pd.columns[0]].isin(c)]

        dirty_pd = dirty_pd.sort([dirty_pd.columns[0]], ascending=[1])
        clean_pd = clean_pd.sort([clean_pd.columns[0]], ascending=[1])

        print dirty_pd.values[:,0]
        print clean_pd.values[:,0]

        super(IQContest, self).__init__("IQContest", dirty_pd, clean_pd)

    def validate(self):
        print "validate"