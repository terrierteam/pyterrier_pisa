from .base import *
import pyterrier as pt
import numpy as np
import pandas as pd

class CiffTest(TempDirTestCase):

    def test_to_and_from_ciff(self):
        from pyterrier_pisa import PisaIndex
        dataset = pt.get_dataset("vaswani")
        idx = PisaIndex(self.test_dir, threads=2)
        idx.index(dataset.get_corpus_iter())
        res_before = idx.bm25().search('chemical reactions')
        idx.to_ciff(self.test_dir + '/ciff')
        idx2 = PisaIndex.from_ciff(self.test_dir + '/ciff', self.test_dir + '/fromciff.pisa')
        res_after = idx2.bm25().search('chemical reactions')
        pd.testing.assert_frame_equal(res_before, res_after)
