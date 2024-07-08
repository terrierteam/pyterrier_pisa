import json
import shutil
from pathlib import Path
import tempfile
import os
import more_itertools
import threading
from warnings import warn
from enum import Enum
from collections import defaultdict
import numpy as np
import ir_datasets
import pyterrier as pt
import pyterrier_pisa
from . import _pisathon

_logger = ir_datasets.log.easy()


class PisaIndexingMode(Enum):
  create = 'create'
  overwrite = 'overwrite'
  # append?

class PisaIndexer(pt.Indexer):
  def __init__(self, path, text_field='text', mode=PisaIndexingMode.create, stemmer='porter2', threads=1, batch_size=100_000):
    self.path = Path(path)
    self.text_field = text_field
    self.mode = PisaIndexingMode(mode)
    self.stemmer = pyterrier_pisa.PisaStemmer(stemmer)
    self.threads = threads
    self.batch_size = batch_size

  def index(self, it):
    path = self.path
    if pyterrier_pisa.PisaIndex.built(self):
      if PisaIndexingMode(self.mode) == PisaIndexingMode.overwrite:
        warn(f'Removing {str(path)}')
        shutil.rmtree(path)
      else:
        raise RuntimeError(f'A PISA index already exists at {path}. If you want to overwrite it, set mode="overwrite"')
    if not path.exists():
      path.mkdir(parents=True, exist_ok=True)

    self._index(it)

    with open(path/'pt_meta.json', 'wt') as fout:
      json.dump({
        'type': 'sparse_index',
        'format': 'pisa',
        'package_hint': 'pyterrier-pisa',
        'stemmer': self.stemmer.value,
      }, fout)
    return pyterrier_pisa.PisaIndex(path, batch_size=self.batch_size, stemmer=self.stemmer, text_field=self.text_field, threads=self.threads)

  def _index(self, it):
    with tempfile.TemporaryDirectory() as d:
      fifo = os.path.join(d, 'fifo')
      os.mkfifo(fifo)
      threading.Thread(target=self._write_fifo, args=(it, fifo, self.text_field), daemon=True).start()
      _pisathon.index(fifo, str(self.path), '' if self.stemmer == pyterrier_pisa.PisaStemmer.none else self.stemmer.value, self.batch_size, self.threads)

  def _write_fifo(self, it, fifo, text_field):
    with open(fifo, 'wt') as fout:
      if isinstance(text_field, str):
        text_field = [text_field]
      for doc in it:
        docno, text = doc['docno'], ' '.join(doc[f] for f in text_field)
        text = text.replace('\n', ' ').replace('\r', ' ') # any other cleanup?
        fout.write(f'{docno} {text}\n')


class PisaToksIndexer(PisaIndexer):
  def __init__(self, path, text_field='toks', mode=PisaIndexingMode.create, threads=1, batch_size=100_000, scale=100.):
    super().__init__(path, text_field, mode, pyterrier_pisa.PisaStemmer.none, threads, batch_size=batch_size)
    self.scale = float(scale)
    assert self.scale > 0

  def _index(self, it):
    lexicon = {}
    docid = 0
    path = self.path
    with (path/'fwd.documents').open('wt') as f_docs, (path/'fwd.terms').open('wt') as f_lex:
      for bidx, batch in enumerate(more_itertools.chunked(it, self.batch_size)):
        _logger.info(f'inverting batch {bidx}: documents [{docid},{docid+len(batch)})')
        inv_did = defaultdict(list)
        inv_score = defaultdict(list)
        lens = []
        for doc in batch:
          l = 0
          f_docs.write(doc['docno']+'\n')
          for term, score in doc[self.text_field].items():
            score = int(score * self.scale)
            if score <= 0:
              continue
            l += score
            if term not in lexicon:
              lexicon[term] = len(lexicon)
              f_lex.write(term+'\n')
            inv_did[lexicon[term]].append(docid)
            inv_score[lexicon[term]].append(int(score))
          lens.append(l)
          docid += 1
        with (path/f'inv.batch.{bidx}.docs').open('wb') as f_did, (path/f'inv.batch.{bidx}.freqs').open('wb') as f_score, (path/f'inv.batch.{bidx}.sizes').open('wb') as f_len:
          f_did.write(np.array([1, len(batch)], dtype=np.uint32).tobytes())
          for i in range(len(lexicon)):
            l = len(inv_did[i])
            f_did.write(np.array([l] + inv_did[i], dtype=np.uint32).tobytes())
            f_score.write(np.array([l] + inv_score[i], dtype=np.uint32).tobytes())
          f_len.write(np.array([len(lens)] + lens, dtype=np.uint32).tobytes())
    _pisathon.merge_inv(str(path/'inv'), bidx+1, len(lexicon))
    for i in range(bidx+1):
      (path/f'inv.batch.{i}.docs').unlink()
      (path/f'inv.batch.{i}.freqs').unlink()
      (path/f'inv.batch.{i}.sizes').unlink()
    (path/'inv.docs').rename(path/'inv.docs.tmp')
    (path/'inv.freqs').rename(path/'inv.freqs.tmp')
    in_docs = np.memmap(path/'inv.docs.tmp', mode='r', dtype=np.uint32)
    in_freqs = np.memmap(path/'inv.freqs.tmp', mode='r', dtype=np.uint32)
    with (path/'fwd.terms').open('wt') as f_lex, (path/'inv.docs').open('wb') as f_docs, (path/'inv.freqs').open('wb') as f_freqs:
      f_docs.write(in_docs[:2].tobytes())
      in_docs = in_docs[2:]
      offsets_lens = []
      i = 0
      while i < in_docs.shape[0]:
        offsets_lens.append((i, in_docs[i]+1))
        i += in_docs[i] + 1
      for term in _logger.pbar(sorted(lexicon), desc='re-mapping term ids'):
        f_lex.write(f'{term}\n')
        i = lexicon[term]
        start, l = offsets_lens[i]
        f_docs.write(in_docs[start:start+l])
        f_freqs.write(in_freqs[start:start+l])
    del in_docs # close mmap
    del in_freqs # close mmap
    (path/'inv.docs.tmp').unlink()
    (path/'inv.freqs.tmp').unlink()
    _pisathon.build_binlex(str(path/'fwd.documents'), str(path/'fwd.doclex'))
    _pisathon.build_binlex(str(path/'fwd.terms'), str(path/'fwd.termlex'))
