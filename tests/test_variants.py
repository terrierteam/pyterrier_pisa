from .base import TempDirTestCase
import pyterrier as pt
import pandas as pd

class VariantTests(TempDirTestCase):

    def test_all(self):
        import pyterrier_pisa
        for encoding in pyterrier_pisa.PisaIndexEncoding:
           self._index_and_retrieve_variant(encoding)

    def _index_and_retrieve_variant(self, encoding):

        from pyterrier_pisa import PisaIndex, PisaQueryAlgorithm
        dataset = pt.get_dataset("vaswani")
        idx = PisaIndex(self.test_dir, index_encoding=encoding, overwrite=True, text_field='text')
        idx.index(dataset.get_corpus_iter())
        for qalg in PisaQueryAlgorithm:
            with self.subTest(encoding=encoding, qalg=qalg):
                TRANS = [
                    idx.bm25(query_algorithm=qalg, precompute_impact=True),
                    idx.bm25(query_algorithm=qalg, precompute_impact=False),
                    idx.dph(query_algorithm=qalg),
                    idx.qld(query_algorithm=qalg),
                    idx.pl2(query_algorithm=qalg)]
                for t in TRANS:
                    results = t.search("chemical reactions")
                    self.assertTrue(len(results) > 0)

if __name__ == '__main__':
  import unittest
  unittest.main()
