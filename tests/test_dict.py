from .base import *
import pyterrier as pt
import numpy as np
import itertools

class DictTest(TempDirTestCase):

    def test_vaswani(self):
      from pyterrier_pisa import PisaIndex, DictTokeniser
      import nltk
      nltk.download('punkt')
      dataset = pt.get_dataset('irds:vaswani')
      idx = PisaIndex(self.test_dir+'/index', text_field='text_toks', stemmer='none')
      idx_pipe = DictTokeniser() >> idx.toks_indexer(scale=1)
      idx_pipe.index(dataset.get_corpus_iter())
      self.assertTrue(idx.built())
      self.assertEqual(len(idx), 11429)
      MAPS=[0.2256]
      TRANS=[DictTokeniser('query') >> idx.bm25()]
      exp_df = pt.Experiment(
          TRANS,
          *dataset.get_topicsqrels(),
          ["map"],
          round=4
      )
      self.assertListEqual(MAPS, exp_df["map"].tolist())

    def test_vaswani_quantized(self):
      from pyterrier_pisa import PisaIndex, DictTokeniser
      import nltk
      nltk.download('punkt')
      dataset = pt.get_dataset('irds:vaswani')
      idx = PisaIndex(self.test_dir+'/index', text_field='text_toks', stemmer='none')
      idx_pipe = DictTokeniser() >> idx.toks_indexer(scale=1)
      idx_pipe.index(dataset.get_corpus_iter())
      self.assertTrue(idx.built())
      self.assertEqual(len(idx), 11429)
      MAPS=[0.0047]
      TRANS=[DictTokeniser('query') >> idx.quantized(num_results=10)]
      res = TRANS[0](dataset.get_topics())
      exp_df = pt.Experiment(
          TRANS,
          *dataset.get_topicsqrels(),
          ["map"],
          round=4
      )
      self.assertListEqual(MAPS, exp_df["map"].tolist())

    def test_dict(self):
        from pyterrier_pisa import PisaIndex
        import pandas as pd
        idx = PisaIndex(self.test_dir+'/index', text_field='text_toks', stemmer='none')
        idx.toks_indexer(scale=1).index([
          {'docno' : 'd1', 'text_toks' : {'a' : 7.3, 'b' : 3.99}}
        ])
        self.assertTrue(idx.built())
        quantized = idx.quantized(toks_scale=1.)
        df_query = pd.DataFrame([['q1', {'a' : 2.3, 'b': 4.1}]], columns=['qid', 'query_toks'])
        res = quantized.transform(df_query)
        self.assertEqual(1, len(res))
        self.assertEqual('d1', res.iloc[0].docno)
        self.assertEqual('q1', res.iloc[0].qid)
        self.assertEqual(26., res.iloc[0].score) # int(7.3) * int(2.3) + int(3.99) * int(4.1) = 7 * 2 + 3 * 4 = 14 + 12 = 

    def test_dict_scale(self):
        from pyterrier_pisa import PisaIndex
        import pandas as pd
        idx = PisaIndex(self.test_dir+'/index', text_field='text_toks', stemmer='none')
        idx.toks_indexer(scale=85.4).index([
          {'docno' : 'd1', 'text_toks' : {'a' : 7.3, 'b' : 3.99}}
        ])
        self.assertTrue(idx.built())
        quantized = idx.quantized(toks_scale=11.2)
        df_query = pd.DataFrame([['q1', {'a' : 2.3, 'b': 4.1}]], columns=['qid', 'query_toks'])
        res = quantized.transform(df_query)
        self.assertEqual(1, len(res))
        self.assertEqual('d1', res.iloc[0].docno)
        self.assertEqual('q1', res.iloc[0].qid)
        self.assertEqual(30875., res.iloc[0].score) # int(7.3 * 85.4) * int(2.3 * 11.2) + int(3.99 * 85.4) * int(4.1 * 11.2) = 33149

if __name__ == "__main__":
  import unittest
  unittest.main()
