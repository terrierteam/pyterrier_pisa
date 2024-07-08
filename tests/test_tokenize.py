from .base import *
import pyterrier as pt
import numpy as np
import itertools

class TestTokenize(TempDirTestCase):

    def test_tokenize(self):
      import pyterrier_pisa
      self.assertListEqual(pyterrier_pisa.tokenize('hello worlds'), ['hello', 'worlds'])
      self.assertListEqual(pyterrier_pisa.tokenize('hello worlds', 'porter2'), ['hello', 'world'])

if __name__ == "__main__":
  import unittest
  unittest.main()
