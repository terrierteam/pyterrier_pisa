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
      idx = PisaIndex(self.test_dir+'/index', text_field='text_dict', stemmer='none')
      idx_pipe = DictTokeniser() >> idx
      idx_pipe.index(dataset.get_corpus_iter())
      self.assertTrue(idx.built())
      self.assertEqual(len(idx), 11429)
      MAPS=[0.2256]
      TRANS=[DictTokeniser('query') >> pt.apply.rename({'query_dict' : 'query_toks'}) >> idx.bm25()]
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
      idx = PisaIndex(self.test_dir+'/index', text_field='text_dict', stemmer='none')
      idx_pipe = DictTokeniser() >> idx
      idx_pipe.index(dataset.get_corpus_iter())
      self.assertTrue(idx.built())
      self.assertEqual(len(idx), 11429)
      MAPS=[0.0047]
      TRANS=[DictTokeniser('query') >> pt.apply.rename({'query_dict' : 'query_toks'}) >> idx.quantized(num_results=10)]
      res = TRANS[0](dataset.get_topics())
      print(res['score'].values.tolist())
      print(res['docno'].values.tolist())
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
        idx = PisaIndex(self.test_dir+'/index', text_field='text_dict', stemmer='none')
        idx.index([
          {'docno' : 'd1', 'text_dict' : {'a' : 1, 'b' : 14}}
        ])
        self.assertTrue(idx.built())
        bm25=idx.bm25()
        df_query = pd.DataFrame([['q1', {'a' : 2.3}]], columns=['qid', 'query_toks'])
        res = bm25.transform(df_query)
        self.assertEqual(1, len(res))
        self.assertEqual('d1', res.iloc[0].docno)
        self.assertEqual('q1', res.iloc[0].qid)
        print(idx.quantized()(pd.DataFrame([['q1', {'a' : 2.3}]], columns=['qid', 'query_toks'])))
        print(idx.quantized()(pd.DataFrame([['q1', {'a' : 1000}]], columns=['qid', 'query_toks'])))
        print(idx.quantized()(pd.DataFrame([['q1', {'a' : 1000, 'b': 1}]], columns=['qid', 'query_toks'])))
        print(idx.quantized()(pd.DataFrame([['q1', {'a' : 1, 'b': 1000}]], columns=['qid', 'query_toks'])))

if __name__ == "__main__":
  import unittest
  unittest.main()
