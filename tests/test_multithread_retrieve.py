from .base import *
import pyterrier as pt

class MultithreadRetrieveTest(TempDirTestCase):

    def test_index_and_retrieve(self):
        from pyterrier_pisa import PisaIndex
        dataset = pt.get_dataset("vaswani")
        idx = PisaIndex(self.test_dir, threads=4)
        idx.index(dataset.get_corpus_iter())

        bm25 = idx.bm25()
        topics = dataset.get_topics()

        for _ in range(10):
            results = bm25.transform(topics)
            assert len(results) > 0

if __name__ == "__main__":
    import unittest
    unittest.main()