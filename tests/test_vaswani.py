from .base import *
import pyterrier as pt
import numpy as np

class VaswaniTest(TempDirTestCase):

    def test_index_and_retrieve(self):
        from pyterrier_pisa import PisaIndex
        dataset = pt.get_dataset("vaswani")
        idx = PisaIndex(self.test_dir, threads=1)
        idx.index(dataset.get_corpus_iter())
        TRANS = [idx.bm25(), idx.dph(), idx.qld(), idx.pl2()]
        MAPS = [0.305349, 0.291841, 0.241206, 0.274852]
        for t in TRANS:
            results = t.search("chemical reactions")
            self.assertEqual(52, len(results))
        exp_df = pt.Experiment(
            TRANS,
            *dataset.get_topicsqrels(),
            ["map"],
            round=6
        )
        self.assertListEqual(MAPS, exp_df["map"].tolist())
        
