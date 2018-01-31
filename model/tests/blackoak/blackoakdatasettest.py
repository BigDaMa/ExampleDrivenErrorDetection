# -*- coding: utf-8 -*-

import unittest

from ml.datasets.blackOak.BlackOakVisualizer import BlackOakVisualizer


class BlackOakDataSetTest(unittest.TestCase):
    """validate dataset"""

    def test_dataset(self):
        blackOakDataSetVi = BlackOakVisualizer()

        blackOakDataSetVi.validate_tools()


if __name__ == '__main__':
    unittest.main()