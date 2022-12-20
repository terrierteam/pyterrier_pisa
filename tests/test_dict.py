from .base import *
import pyterrier as pt
import numpy as np
import itertools

class DictTest(TempDirTestCase):

  # def test_dict(self):
  #       from pyterrier_pisa import PisaIndex, DictTokeniser
  #       dataset = pt.get_dataset('irds:msmarco-passage')
  #       idx = PisaIndex(self.test_dir+'/index', text_field='text_dict', stemmer='none')
  #       #(DictTokeniser() >> idx).index(itertools.islice(dataset.get_corpus_iter(), 200000))
  #       #(DictTokeniser() >> idx).index(dataset.get_corpus_iter())
  #       idx.index(PisaIndex('/home/sean/data/indices/msmarco-passage.pisa').get_corpus_iter())
  #       import pdb; pdb.set_trace()
  #       pass

    def test_vaswani(self):
      from pyterrier_pisa import PisaIndex, DictTokeniser
      import nltk
      nltk.download('punkt')
      dataset = pt.get_dataset('irds:vaswani')
      idx = PisaIndex(self.test_dir+'/index', text_field='text_dict', pretokenised=True, stemmer='none')
      idx_pipe = DictTokeniser() >> idx
      idx_pipe.index(dataset.get_corpus_iter())
      self.assertTrue(idx.built())
      self.assertEqual(len(idx), 11429)
      MAPS=[0.225637]
      TRANS=[DictTokeniser('query') >> pt.apply.rename({'query_dict' : 'query_toks'}) >> idx.bm25()]
      TRANS[0].pretokenised = False
      exp_df = pt.Experiment(
          TRANS,
          *dataset.get_topicsqrels(),
          ["map"],
          round=6
      )
      self.assertListEqual(MAPS, exp_df["map"].tolist())

    def test_dict(self):
        from pyterrier_pisa import PisaIndex
        import pandas as pd
        idx = PisaIndex(self.test_dir+'/index', text_field='text_dict', pretokenised=True, stemmer='none')
        idx.index([
          {'docno' : 'd1', 'text_dict' : {'a' : 1, 'b' : 14}}
        ])
        self.assertTrue(idx.built())
        bm25=idx.bm25()
        bm25.pretokenised = True
        df_query = pd.DataFrame([['q1', {'a' : 2.3}]], columns=['qid', 'query_toks'])
        res = bm25.transform(df_query)
        self.assertEqual(1, len(res))
        self.assertEqual('d1', res.iloc[0].docno)
        self.assertEqual('q1', res.iloc[0].qid)

        #import pdb; pdb.set_trace()
        #pass

if __name__ == "__main__":
  import unittest
  unittest.main()
