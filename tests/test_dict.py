from .base import *
import pyterrier as pt
import numpy as np
import itertools

class DictTest(TempDirTestCase):

    def test_dict(self):
        from pyterrier_pisa import PisaIndex, DictTokeniser
        dataset = pt.get_dataset('irds:msmarco-passage')
        idx = PisaIndex(self.test_dir+'/index', text_field='text_dict', stemmer='none')
        #(DictTokeniser() >> idx).index(itertools.islice(dataset.get_corpus_iter(), 200000))
        #(DictTokeniser() >> idx).index(dataset.get_corpus_iter())
        idx.index(PisaIndex('/home/sean/data/indices/msmarco-passage.pisa').get_corpus_iter())
        import pdb; pdb.set_trace()
        pass

if __name__ == "__main__":
  import unittest
  unittest.main()
