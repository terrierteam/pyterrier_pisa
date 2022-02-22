import numpy as np
from . import _pisathon
import json
import shutil
import sys
import pandas as pd
from pathlib import Path
import tempfile
import os
import more_itertools
import threading
from warnings import warn
from typing import Optional, Union, List
from enum import Enum
from collections import Counter
import pyterrier as pt
from pyterrier.datasets import Dataset
import functools


class PisaStemmer(Enum):
  """
  Represents a built-in stemming function from PISA
  """
  none = 'none'
  porter2 = 'porter2'
  krovetz = 'krovetz'


class PisaScorer(Enum):
  """
  Represents a built-in scoring function from PISA
  """
  bm25 = 'bm25'
  dph = 'dph'
  pl2 = 'pl2'
  qld = 'qld'


class PisaIndexEncoding(Enum):
  """
  Represents a built-in index encoding type from PISA.
  """
  ef = 'ef'
  single = 'single'
  pefuniform = 'pefuniform'
  pefopt = 'pefopt'
  block_optpfor = 'block_optpfor'
  block_varintg8iu = 'block_varintg8iu'
  block_streamvbyte = 'block_streamvbyte'
  block_maskedvbyte = 'block_maskedvbyte'
  block_interpolative = 'block_interpolative'
  block_qmx = 'block_qmx'
  block_varintgb = 'block_varintgb'
  block_simple8b = 'block_simple8b'
  block_simple16 = 'block_simple16'
  block_simdbp = 'block_simdbp'


class PisaQueryAlgorithm(Enum):
  """
  Represents a built-in query algorithm
  """
  wand = 'wand'
  block_max_wand = 'block_max_wand'
  block_max_maxscore = 'block_max_maxscore'
  block_max_ranked_and = 'block_max_ranked_and'
  ranked_and = 'ranked_and'
  ranked_or = 'ranked_or'
  maxscore = 'maxscore'


class PisaStopwords(Enum):
  """
  Represents which set of stopwords to use during retrieval
  """
  terrier = 'terrier'
  none = 'none'


PISA_INDEX_DEFAULTS = {
  'stemmer': PisaStemmer.porter2,
  'index_encoding': PisaIndexEncoding.block_simdbp,
  'query_algorithm': PisaQueryAlgorithm.block_max_wand,
  'stops': PisaStopwords.terrier,
}


def log_level(on=True):
  _pisathon.log_level(1 if on else 0)


class PisaIndex(pt.transformer.IterDictIndexerBase):
  def __init__(self,
      path: str,
      text_field: str = None,
      stemmer: Optional[Union[PisaStemmer, str]] = None,
      index_encoding: Optional[Union[PisaIndexEncoding, str]] = None,
      batch_size: int = 100_000,
      stops: Optional[Union[PisaStopwords, List[str]]] = None,
      threads: int = 8,
      overwrite=False):
    self.path = path
    ppath = Path(path)
    if stemmer is not None: stemmer = PisaStemmer(stemmer)
    if index_encoding is not None: index_encoding = PisaIndexEncoding(index_encoding)
    if stops is not None and not isinstance(stops, list): stops = PisaStopwords(stops)
    if (ppath/'pt_pisa_config.json').exists():
      with (ppath/'pt_pisa_config.json').open('rt') as fin:
        config = json.load(fin)
      if stemmer is None:
        stemmer = PisaStemmer(config['stemmer'])
      if stemmer.value != config['stemmer']:
        warn(f'requested stemmer={stemmer.value}, but index was constructed with {config["stemmer"]}')
    if stemmer is None: stemmer = PISA_INDEX_DEFAULTS['stemmer']
    if index_encoding is None: index_encoding = PISA_INDEX_DEFAULTS['index_encoding']
    if stops is None: stops = PISA_INDEX_DEFAULTS['stops']
    self.text_field = text_field
    self.stemmer = stemmer
    self.index_encoding = index_encoding
    self.batch_size = batch_size
    self.threads = threads
    self.overwrite = overwrite
    self.stops = stops

  def transform(self, *args, **kwargs):
    raise RuntimeError(f'You cannot use {self} itself as a transformer. Did you mean to call a ranking function like .bm25()?')

  def built(self):
    return (Path(self.path)/'pt_pisa_config.json').exists()

  def index(self, inp):
    with tempfile.TemporaryDirectory() as d:
      fifo = os.path.join(d, 'fifo')
      os.mkfifo(fifo)
      threading.Thread(target=self._write_fifo, args=(inp, fifo), daemon=True).start()
      ppath = Path(self.path)
      if self.built():
        if self.overwrite:
          warn(f'Removing {str(ppath)}')
          shutil.rmtree(ppath)
        else:
          raise RuntimeError('A PISA index already exists at {self.path}. If you want to overwrite it, pass overwrite=True to PisaIndex.')
      if not ppath.exists():
        ppath.mkdir(parents=True, exist_ok=True)
      _pisathon.index(fifo, self.path, '' if self.stemmer == PisaStemmer.none else self.stemmer.value, self.batch_size, self.threads)
      with open(ppath/'pt_pisa_config.json', 'wt') as fout:
        json.dump({
          'stemmer': self.stemmer.value,
        }, fout)

  def _write_fifo(self, it, fifo):
    with open(fifo, 'wt') as fout:
      text_field = self.text_field
      if text_field is None:
        it = more_itertools.peekable(it)
        first_doc = it.peek()
        text_field = [k for k, v in sorted(first_doc.items()) if isinstance(v, str) and k != 'docno']
        assert len(text_field) >= 1, f"no str fields found in document. Fields: {k: type(v) for k, v in first_doc.items()}"
        warn(f'text_field not specified; indexing str fields found in the first document: {text_field}')
      elif isinstance(text_field, str):
        text_field = [text_field]
      for doc in it:
        docno, text = doc['docno'], ' '.join(doc[f] for f in text_field)
        text = text.replace('\n', ' ').replace('\r', ' ') # any other cleanup?
        fout.write(f'{docno} {text}\n')

  def bm25(self, k1=0.9, b=0.4, num_results=1000, verbose=False, threads=None, query_algorithm=None):
    return PisaRetrieve(self, scorer=PisaScorer.bm25, bm25_k1=k1, bm25_b=b, num_results=num_results, verbose=verbose, threads=threads or self.threads, stops=self.stops, query_algorithm=query_algorithm)

  def dph(self, num_results=1000, verbose=False, threads=None, query_algorithm=None):
    return PisaRetrieve(self, scorer=PisaScorer.dph, num_results=num_results, verbose=verbose, threads=threads or self.threads, stops=self.stops, query_algorithm=query_algorithm)

  def pl2(self, c=1., num_results=1000, verbose=False, threads=None, query_algorithm=None):
    return PisaRetrieve(self, scorer=PisaScorer.pl2, pl2_c=c, num_results=num_results, verbose=verbose, threads=threads or self.threads, stops=self.stops, query_algorithm=query_algorithm)

  def qld(self, mu=1000., num_results=1000, verbose=False, threads=None, query_algorithm=None):
    return PisaRetrieve(self, scorer=PisaScorer.qld, qld_mu=mu, num_results=num_results, verbose=verbose, threads=threads or self.threads, stops=self.stops, query_algorithm=query_algorithm)

  def num_terms(self):
    if self.built():
      return _pisathon.num_terms(self.path)

  def num_docs(self):
    if self.built():
      return _pisathon.num_docs(self.path)

  def __repr__(self):
    return f'PisaIndex({repr(self.path)})'

  @staticmethod
  def from_dataset(dataset: Union[str, Dataset], variant: str = 'pisa_porter2', version: str = 'latest', **kwargs):
    from pyterrier.batchretrieve import _from_dataset
    return _from_dataset(dataset, variant=variant, version=version, clz=PisaIndex, **kwargs)


class PisaRetrieve(pt.transformer.TransformerBase):
  def __init__(self, index: Union[PisaIndex, str], scorer: Union[PisaScorer, str], num_results: int = 1000, threads=None, verbose=False, stops=None, query_algorithm=None, **retr_args):
    if isinstance(index, PisaIndex):
      self.index = index
    else:
      self.index = PisaIndex(index)
    assert self.index.built(), f"Index at {self.index.path} is not built. Before you can use it for retrieval, you need to index."
    self.scorer = PisaScorer(scorer)
    self.num_results = num_results
    self.retr_args = retr_args
    self.verbose = verbose
    self.threads = threads or self.index.threads
    if stops is None:
      stpps = self.index.stops
    self.stops = PisaStopwords(stops)
    if query_algorithm is None:
      query_algorithm = PISA_INDEX_DEFAULTS['query_algorithm']
    self.query_algorithm = PisaQueryAlgorithm(query_algorithm)
    _pisathon.prepare_index(self.index.path, encoding=self.index.index_encoding.value, scorer_name=self.scorer.value, **retr_args)

  def transform(self, queries):
    inp = []
    mapping = {}
    for i, q in enumerate(queries.itertuples(index=False)):
      qid = str(q.qid)
      inp.append((qid, q.query))
      if qid in mapping:
        raise ValueError(f'duplicate qids detected (e.g., {repr(qid)})')
      mapping[qid] = i
    if self.verbose:
      inp = pt.tqdm(inp, unit='query', desc=f'PISA {self.scorer.value}')
    with tempfile.TemporaryDirectory() as d:
      qids, docnos, ranks, scores = _pisathon.retrieve(
        self.index.path,
        self.index.index_encoding.value,
        self.query_algorithm.value,
        self.scorer.value,
        '' if self.index.stemmer == PisaStemmer.none else self.index.stemmer.value,
        inp,
        k=self.num_results,
        threads=self.threads,
        stop_fname=self._stops_fname(d),
        **self.retr_args)
    idxs = np.vectorize(mapping.__getitem__, otypes=[np.int32])(qids)
    df = {'qid': qids, 'docno': docnos, 'rank': ranks, 'score': scores}
    df.update({c: queries[c].iloc[idxs] for c in queries.columns if c != 'qid'})
    return pd.DataFrame(df)

  def __repr__(self):
    return f'PisaRetrieve({repr(self.index)}, {repr(self.scorer)}, {repr(self.num_results)}, {repr(self.retr_args)}, {repr(self.stops)})'

  @staticmethod
  def from_dataset(dataset: Union[str, Dataset], variant: str = None, version: str = 'latest', **kwargs):
    from pyterrier.batchretrieve import _from_dataset
    return _from_dataset(dataset, variant=variant, version=version, clz=PisaRetrieve, **kwargs)

  def _stops_fname(self, d):
    if self.stops == PisaStopwords.none:
      return ''
    else:
      fifo = os.path.join(d, 'stops')
      stops = self.stops
      if stops == PisaStopwords.terrier:
        stops = _terrier_stops()
      with open(fifo, 'wt') as fout:
        for stop in stops:
          fout.write(f'{stop}\n')
      return fifo

@functools.lru_cache()
def _terrier_stops():
  Stopwords = pt.autoclass('org.terrier.terms.Stopwords')
  stops = list(Stopwords(None).stopWords)
  return stops

def main():
  if not pt.started():
    pt.init()
  import argparse
  parser = argparse.ArgumentParser('pyterrier_pisa')
  parser.set_defaults(func=lambda x: parser.print_help())
  subparsers = parser.add_subparsers()
  index_parser = subparsers.add_parser('index')
  index_parser.add_argument('index_path')
  index_parser.add_argument('dataset')
  index_parser.add_argument('--threads', type=int, default=8)
  index_parser.add_argument('--batch_size', type=int, default=100_000)
  index_parser.add_argument('--fields', nargs='+')
  index_parser.set_defaults(func=main_index)

  retrieve_parser = subparsers.add_parser('retrieve')
  retrieve_parser.add_argument('index_path')
  retrieve_parser.add_argument('dataset')
  retrieve_parser.add_argument('scorer', choices=PisaScorer.__members__.values(), type=PisaScorer)
  retrieve_parser.add_argument('--num_results', '-k', type=int, default=1000)
  retrieve_parser.add_argument('--stops', choices=PisaStopwords.__members__.values(), type=PisaStopwords, default=PISA_INDEX_DEFAULTS['stops'])
  retrieve_parser.add_argument('--field', default=None)
  retrieve_parser.add_argument('--threads', type=int, default=8)
  retrieve_parser.add_argument('--batch_size', type=int, default=None)
  retrieve_parser.set_defaults(func=main_retrieve)

  args = parser.parse_args()
  args.func(args)


def main_index(args):
  dataset = pt.get_dataset(args.dataset)
  index = PisaIndex(args.index_path, args.fields, threads=args.threads, batch_size=args.batch_size)
  docs = dataset.get_corpus_iter(verbose=False)
  total = None
  if hasattr(dataset, 'irds_ref'):
    total = dataset.irds_ref().docs_count()
  docs = pt.tqdm(docs, total=total, smoothing=0., desc='document feed', unit='doc')
  index.index(docs)


def main_retrieve(args):
  dataset = pt.get_dataset(args.dataset)
  index = PisaIndex(args.index_path)
  retr = PisaRetrieve(index, args.scorer, num_results=args.num_results, stops=args.stops, threads=args.threads)
  topics = dataset.get_topics(args.field)
  if args.batch_size:
    it = pt.tqdm(range(0, len(topics), args.batch_size))
  else:
    it = [None]
  for i in it:
    if args.batch_size:
      batch = topics.iloc[i:i+args.batch_size]
    else:
      batch = topics
    res = retr(batch)
    for r in res.itertuples(index=False):
      sys.stdout.write(f'{r.qid} 0 {r.docno} {r.rank} {r.score:.4f} pt_pisa\n')

if __name__ == '__main__':
  main()
