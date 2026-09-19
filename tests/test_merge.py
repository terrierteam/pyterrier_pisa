from .base import *
import pyterrier as pt
import numpy as np
import pandas as pd
import itertools


BINARY_FILES = ['inv.docs', 'inv.freqs', 'inv.sizes', 'fwd', 'fwd.terms', 'fwd.documents', 'fwd.termlex', 'fwd.doclex']


def _assert_same_files(test_case, dir_a, dir_b, files=BINARY_FILES):
    for fname in files:
        a = dir_a / fname
        b = dir_b / fname
        test_case.assertEqual(a.exists(), b.exists(), f'{fname} exists mismatch between {dir_a} and {dir_b}')
        if a.exists():
            test_case.assertEqual(a.read_bytes(), b.read_bytes(), f'{fname} differs between {dir_a} and {dir_b}')


class MergeTest(TempDirTestCase):

    def test_equivalence_plaintext(self):
        from pyterrier_pisa import PisaIndex
        from pathlib import Path
        dataset = pt.get_dataset("vaswani")
        corpus = list(dataset.get_corpus_iter())
        half = len(corpus) // 2

        baseline_path = Path(self.test_dir) / 'baseline'
        idx = PisaIndex(str(baseline_path))
        idx.index(iter(corpus))

        a_path = Path(self.test_dir) / 'a'
        b_path = Path(self.test_dir) / 'b'
        idx_a = PisaIndex(str(a_path))
        idx_a.index(iter(corpus[:half]))
        idx_b = PisaIndex(str(b_path))
        idx_b.index(iter(corpus[half:]))

        out_path = Path(self.test_dir) / 'merged'
        merged = PisaIndex.merge([idx_a, idx_b], str(out_path))

        self.assertEqual(idx.num_docs(), merged.num_docs())
        self.assertEqual(idx.num_terms(), merged.num_terms())
        _assert_same_files(self, baseline_path, out_path)

        res_before = idx.bm25().search('chemical reactions')
        res_after = merged.bm25().search('chemical reactions')
        pd.testing.assert_frame_equal(res_before, res_after)

    def test_equivalence_toks(self):
        from pyterrier_pisa import PisaIndex, DictTokeniser
        from pathlib import Path
        import nltk
        nltk.download('punkt')
        dataset = pt.get_dataset('irds:vaswani')
        corpus = list(DictTokeniser().transform(pd.DataFrame(dataset.get_corpus_iter())).to_dict('records'))
        half = len(corpus) // 2

        baseline_path = Path(self.test_dir) / 'baseline'
        idx = PisaIndex(str(baseline_path), text_field='text_toks', stemmer='none')
        idx.toks_indexer(scale=1).index(iter(corpus))

        a_path = Path(self.test_dir) / 'a'
        b_path = Path(self.test_dir) / 'b'
        idx_a = PisaIndex(str(a_path), text_field='text_toks', stemmer='none')
        idx_a.toks_indexer(scale=1).index(iter(corpus[:half]))
        idx_b = PisaIndex(str(b_path), text_field='text_toks', stemmer='none')
        idx_b.toks_indexer(scale=1).index(iter(corpus[half:]))

        out_path = Path(self.test_dir) / 'merged'
        merged = PisaIndex.merge([idx_a, idx_b], str(out_path))

        self.assertEqual(idx.num_docs(), merged.num_docs())
        self.assertEqual(idx.num_terms(), merged.num_terms())
        self.assertFalse((out_path / 'fwd').exists())  # toks indexes never have a raw fwd
        _assert_same_files(self, baseline_path, out_path, files=[f for f in BINARY_FILES if f != 'fwd'])

    def test_single_input(self):
        from pyterrier_pisa import PisaIndex
        from pathlib import Path
        a_path = Path(self.test_dir) / 'a'
        idx_a = PisaIndex(str(a_path))
        idx_a.index(iter([{'docno': f'd{i}', 'text': f'term{i} shared'} for i in range(20)]))

        out_path = Path(self.test_dir) / 'out'
        merged = PisaIndex.merge([idx_a], str(out_path))
        self.assertEqual(idx_a.num_docs(), merged.num_docs())
        _assert_same_files(self, a_path, out_path)

    def test_self_merge(self):
        from pyterrier_pisa import PisaIndex
        from pathlib import Path
        a_path = Path(self.test_dir) / 'a'
        idx_a = PisaIndex(str(a_path))
        idx_a.index(iter([{'docno': f'd{i}', 'text': f'term{i} shared'} for i in range(20)]))

        out_path = Path(self.test_dir) / 'out'
        merged = PisaIndex.merge([idx_a, idx_a], str(out_path))
        self.assertEqual(2 * idx_a.num_docs(), merged.num_docs())
        res = merged.bm25(num_results=100).search('shared')
        # every docno should appear exactly twice, with equal scores
        counts = res['docno'].value_counts()
        self.assertTrue((counts == 2).all())

    def test_disjoint_vocab(self):
        from pyterrier_pisa import PisaIndex
        from pathlib import Path
        a_path = Path(self.test_dir) / 'a'
        b_path = Path(self.test_dir) / 'b'
        idx_a = PisaIndex(str(a_path), stemmer='none')
        idx_a.toks_indexer().index(iter([{'docno': 'a0', 'toks': {'onlya': 1.0, 'shared': 1.0}}]))
        idx_b = PisaIndex(str(b_path), stemmer='none')
        idx_b.toks_indexer().index(iter([{'docno': 'b0', 'toks': {'onlyb': 1.0, 'shared': 1.0}}]))

        out_path = Path(self.test_dir) / 'out'
        merged = PisaIndex.merge([idx_a, idx_b], str(out_path))
        self.assertEqual(3, merged.num_terms())  # onlya, onlyb, shared
        self.assertEqual(2, merged.num_docs())

        res = merged.bm25(num_results=10).search('onlya')
        self.assertListEqual(['a0'], res['docno'].tolist())
        res = merged.bm25(num_results=10).search('onlyb')
        self.assertListEqual(['b0'], res['docno'].tolist())

    def test_overwrite_clears_derived(self):
        from pyterrier_pisa import PisaIndex
        from pathlib import Path
        a_path = Path(self.test_dir) / 'a'
        b_path = Path(self.test_dir) / 'b'
        idx_a = PisaIndex(str(a_path))
        idx_a.index(iter([{'docno': 'a0', 'text': 'hello world'}]))
        idx_b = PisaIndex(str(b_path))
        idx_b.index(iter([{'docno': 'b0', 'text': 'goodbye world'}]))

        out_path = Path(self.test_dir) / 'out'
        merged = PisaIndex.merge([idx_a], str(out_path))
        merged.bm25().search('hello')  # trigger building of cached compressed/wand structures
        self.assertTrue(any('.bmw.' in f.name for f in out_path.iterdir()))

        merged2 = PisaIndex.merge([idx_b], str(out_path), overwrite=True)
        self.assertFalse(any('.bmw.' in f.name for f in out_path.iterdir()))
        res = merged2.bm25().search('goodbye')
        self.assertListEqual(['b0'], res['docno'].tolist())

    def test_validation_errors(self):
        from pyterrier_pisa import PisaIndex
        from pathlib import Path
        a_path = Path(self.test_dir) / 'a'
        idx_a = PisaIndex(str(a_path))
        idx_a.index(iter([{'docno': 'a0', 'text': 'hello world'}]))

        with self.assertRaises(ValueError):
            PisaIndex.merge([], str(Path(self.test_dir) / 'out'))

        with self.assertRaises(ValueError):
            PisaIndex.merge([idx_a], str(a_path))  # output == input

        with self.assertRaises(ValueError):
            PisaIndex.merge([idx_a], str(Path(self.test_dir)))  # output contains input

        stemmed_path = Path(self.test_dir) / 'stemmed'
        idx_stemmed = PisaIndex(str(stemmed_path), stemmer='porter2')
        idx_stemmed.index(iter([{'docno': 's0', 'text': 'running'}]))
        none_path = Path(self.test_dir) / 'none'
        idx_none = PisaIndex(str(none_path), stemmer='none')
        idx_none.index(iter([{'docno': 'n0', 'text': 'running'}]))
        with self.assertRaises(ValueError):
            PisaIndex.merge([idx_stemmed, idx_none], str(Path(self.test_dir) / 'out2'))
        # force=True proceeds anyway
        forced = PisaIndex.merge([idx_stemmed, idx_none], str(Path(self.test_dir) / 'out3'), force=True)
        self.assertEqual(2, forced.num_docs())

        toks_path = Path(self.test_dir) / 'toks'
        idx_toks = PisaIndex(str(toks_path), stemmer='none')
        idx_toks.toks_indexer().index(iter([{'docno': 't0', 'toks': {'z': 1.0}}]))
        with self.assertRaises(ValueError):
            PisaIndex.merge([idx_none, idx_toks], str(Path(self.test_dir) / 'out4'))

        PisaIndex.merge([idx_a], str(Path(self.test_dir) / 'out5'))
        with self.assertRaises(RuntimeError):
            PisaIndex.merge([idx_a], str(Path(self.test_dir) / 'out5'))  # exists, no overwrite


if __name__ == "__main__":
    import unittest
    unittest.main()
