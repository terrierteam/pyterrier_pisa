import json
import shutil
from pathlib import Path
import tempfile
import os
import more_itertools
import threading
from warnings import warn
from enum import Enum
from collections import defaultdict, Counter
import numpy as np
import ir_datasets
import pyterrier as pt
import pyterrier_pisa
from . import _pisathon

_logger = ir_datasets.log.easy()


class PisaIndexingMode(Enum):
  create = 'create'
  overwrite = 'overwrite'
  append = 'append'

class PisaIndexer(pt.Indexer):
  def __init__(self, path, text_field='text', mode=PisaIndexingMode.create, stemmer='porter2', threads=1, batch_size=100_000):
    self.path = Path(path)
    self.text_field = text_field
    self.mode = PisaIndexingMode(mode)
    self.stemmer = pyterrier_pisa.PisaStemmer(stemmer)
    self.threads = threads
    self.batch_size = batch_size

  def index_inputs(self):
    if isinstance(self.text_field, str):
      return [["docno", self.text_field]]
    return [["docno"] + list(self.text_field)]

  def index(self, it):
    path = self.path
    exists = pyterrier_pisa.PisaIndex.built(self)
    if exists:
      if self.mode == PisaIndexingMode.overwrite:
        warn(f'Removing {str(path)}')
        shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
      elif self.mode == PisaIndexingMode.create:
        raise RuntimeError(f'A PISA index already exists at {path}. If you want to overwrite it, set mode="overwrite"')
      # else: mode == append -- keep the existing index files where they are
    else:
      if self.mode == PisaIndexingMode.append:
        raise RuntimeError(f'No index exists at {path} to append to. Use mode="create" to build a new index.')
      path.mkdir(parents=True, exist_ok=True)

    if self.mode == PisaIndexingMode.append:
      self._append(it)
      _finalize_append(path)
    else:
      self._index(it)

    with open(path/'pt_meta.json', 'wt') as fout:
      json.dump({
        'type': 'sparse_index',
        'format': 'pisa',
        'package_hint': 'pyterrier-pisa',
        'stemmer': self.stemmer.value,
        'indexer': self._indexer_kind(),
      }, fout)
    return pyterrier_pisa.PisaIndex(path, batch_size=self.batch_size, stemmer=self.stemmer, text_field=self.text_field, threads=self.threads)

  def _indexer_kind(self):
    return 'text'

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

  def _doc_toks(self, doc):
    """Tokenizes a document's text field(s) into a term->count dict. Used for batch-based
    (re)indexing, i.e. append, since that path can't use the direct C++ FIFO/forward-index
    builder (which assumes a fresh index)."""
    text_field = self.text_field
    if isinstance(text_field, str):
      text_field = [text_field]
    text = ' '.join(doc[f] for f in text_field)
    text = text.replace('\n', ' ').replace('\r', ' ')
    stemmer = '' if self.stemmer == pyterrier_pisa.PisaStemmer.none else self.stemmer.value
    return Counter(_pisathon.tokenize(text, stemmer))

  def _append(self, it):
    path = self.path
    for fname in ('inv.docs', 'inv.freqs', 'inv.sizes'):
      if not (path/fname).exists():
        raise RuntimeError(
          f'Cannot append: {path/fname} not found. The index may be too old, incomplete, or '
          'built by a tool that does not produce this file.')

    with (path/'pt_meta.json').open('rt') as fin:
      config = json.load(fin)
    if config.get('stemmer') != self.stemmer.value:
      raise RuntimeError(
        f'Cannot append with stemmer={self.stemmer.value!r}; index at {path} was built with '
        f'stemmer={config.get("stemmer")!r}. Construct the indexer with the matching stemmer.')
    expected_kind = self._indexer_kind()
    existing_kind = config.get('indexer')
    if existing_kind is None:
      warn(
        f'Index at {path} does not record which indexer built it (it may predate append support, '
        f'or have been imported, e.g. via from_ciff); proceeding with append using a '
        f'{expected_kind!r} indexer, but mixing indexing methods on the same index can silently '
        'produce inconsistent scores.')
    elif existing_kind != expected_kind:
      raise RuntimeError(
        f'Cannot append using a {expected_kind!r} indexer to an index at {path} that was built '
        f'with a {existing_kind!r} indexer -- mixing indexing methods would silently produce '
        'inconsistent scores.')

    # Load the existing lexicon (in on-disk/term-id order) and docnos (to reject duplicates).
    # Both are read fully into memory, same as e.g. PisaIndex.get_corpus_iter already does for
    # the term lexicon.
    lexicon = {}
    with (path/'fwd.terms').open('rt') as f_lex:
      for term in f_lex:
        term = term.rstrip('\n')
        lexicon[term] = len(lexicon)
    old_term_count = len(lexicon)
    docnos = set()
    with (path/'fwd.documents').open('rt') as f_docs:
      for docno in f_docs:
        docnos.add(docno.rstrip('\n'))
    docid = _pisathon.num_docs(str(path))

    # Record current sizes so a failure partway through the batch loop below can be rolled back.
    docs_offset = (path/'fwd.documents').stat().st_size
    terms_offset = (path/'fwd.terms').stat().st_size

    # Stage the existing (already-merged & remapped) inverted index as "batch 0". Its on-disk
    # format is identical to a batch file's (pisa::invert::merge_batches produces the same
    # sequence-per-term layout it consumes), so it can be merged with newly-written batches
    # covering the appended documents.
    (path/'inv.docs').rename(path/'inv.batch.0.docs')
    (path/'inv.freqs').rename(path/'inv.batch.0.freqs')
    (path/'inv.sizes').rename(path/'inv.batch.0.sizes')
    batch_term_counts = {0: old_term_count}
    bidx = 0

    try:
      with (path/'fwd.documents').open('at') as f_docs, (path/'fwd.terms').open('at') as f_lex:
        for bidx, batch in enumerate(more_itertools.chunked(it, self.batch_size), start=1):
          _logger.info(f'appending batch {bidx}: documents [{docid},{docid+len(batch)})')
          inv_did = defaultdict(list)
          inv_score = defaultdict(list)
          lens = []
          for doc in batch:
            docno = doc['docno']
            if docno in docnos:
              raise ValueError(f'Cannot append: docno {docno!r} already exists in the index at {path}')
            docnos.add(docno)
            doclen = 0
            f_docs.write(docno+'\n')
            for term, score in self._doc_toks(doc).items():
              score = int(score)
              if score <= 0:
                continue
              doclen += score
              if term not in lexicon:
                lexicon[term] = len(lexicon)
                f_lex.write(term+'\n')
              inv_did[lexicon[term]].append(docid)
              inv_score[lexicon[term]].append(score)
            lens.append(doclen)
            docid += 1
          batch_term_counts[bidx] = len(lexicon)
          with (path/f'inv.batch.{bidx}.docs').open('wb') as f_did, (path/f'inv.batch.{bidx}.freqs').open('wb') as f_score, (path/f'inv.batch.{bidx}.sizes').open('wb') as f_len:
            f_did.write(np.array([1, len(batch)], dtype=np.uint32).tobytes())
            for i in range(len(lexicon)):
              doclen = len(inv_did[i])
              f_did.write(np.array([doclen] + inv_did[i], dtype=np.uint32).tobytes())
              f_score.write(np.array([doclen] + inv_score[i], dtype=np.uint32).tobytes())
            f_len.write(np.array([len(lens)] + lens, dtype=np.uint32).tobytes())
    except BaseException:
      # Undo the batch-0 staging and any newly-written batches/appended docnos/terms, so a
      # failure partway through (bad input, a duplicate docno, an interrupt) leaves the index
      # exactly as it was before this call rather than half-migrated.
      with (path/'fwd.documents').open('r+b') as f:
        f.truncate(docs_offset)
      with (path/'fwd.terms').open('r+b') as f:
        f.truncate(terms_offset)
      for i in range(1, bidx + 1):
        for suffix in ('docs', 'freqs', 'sizes'):
          fp = path/f'inv.batch.{i}.{suffix}'
          if fp.exists():
            fp.unlink()
      (path/'inv.batch.0.docs').rename(path/'inv.docs')
      (path/'inv.batch.0.freqs').rename(path/'inv.freqs')
      (path/'inv.batch.0.sizes').rename(path/'inv.sizes')
      raise

    if bidx == 0:
      # no new documents were provided -- undo the batch-0 staging and leave the index untouched
      (path/'inv.batch.0.docs').rename(path/'inv.docs')
      (path/'inv.batch.0.freqs').rename(path/'inv.freqs')
      (path/'inv.batch.0.sizes').rename(path/'inv.sizes')
      return

    # From here on, only deterministic file operations over data already written to disk remain
    # (no more user-supplied documents flow through) -- not wrapped in the rollback above.
    _pad_and_merge_batches(path, batch_term_counts, len(lexicon))
    _remap_terms_alphabetically(path, lexicon)


class PisaToksIndexer(PisaIndexer):
  def __init__(self, path, text_field='toks', mode=PisaIndexingMode.create, threads=1, batch_size=100_000, scale=100.):
    super().__init__(path, text_field, mode, pyterrier_pisa.PisaStemmer.none, threads, batch_size=batch_size)
    self.scale = float(scale)
    assert self.scale > 0

  def _doc_toks(self, doc):
    return {term: int(score * self.scale) for term, score in doc[self.text_field].items()}

  def _indexer_kind(self):
    return 'toks'

  def _index(self, it):
    lexicon = {}
    docid = 0
    path = self.path
    batch_term_counts = {}
    with (path/'fwd.documents').open('wt') as f_docs, (path/'fwd.terms').open('wt') as f_lex:
      for bidx, batch in enumerate(more_itertools.chunked(it, self.batch_size)):
        _logger.info(f'inverting batch {bidx}: documents [{docid},{docid+len(batch)})')
        inv_did = defaultdict(list)
        inv_score = defaultdict(list)
        lens = []
        for doc in batch:
          doclen = 0
          f_docs.write(doc['docno']+'\n')
          for term, score in self._doc_toks(doc).items():
            score = int(score)
            if score <= 0:
              continue
            doclen += score
            if term not in lexicon:
              lexicon[term] = len(lexicon)
              f_lex.write(term+'\n')
            inv_did[lexicon[term]].append(docid)
            inv_score[lexicon[term]].append(score)
          lens.append(doclen)
          docid += 1
        batch_term_counts[bidx] = len(lexicon)
        with (path/f'inv.batch.{bidx}.docs').open('wb') as f_did, (path/f'inv.batch.{bidx}.freqs').open('wb') as f_score, (path/f'inv.batch.{bidx}.sizes').open('wb') as f_len:
          f_did.write(np.array([1, len(batch)], dtype=np.uint32).tobytes())
          for i in range(len(lexicon)):
            doclen = len(inv_did[i])
            f_did.write(np.array([doclen] + inv_did[i], dtype=np.uint32).tobytes())
            f_score.write(np.array([doclen] + inv_score[i], dtype=np.uint32).tobytes())
          f_len.write(np.array([len(lens)] + lens, dtype=np.uint32).tobytes())
    _pad_and_merge_batches(path, batch_term_counts, len(lexicon))
    _remap_terms_alphabetically(path, lexicon)


def _pad_and_merge_batches(path, batch_term_counts, final_term_count):
  """Merges a set of `inv.batch.{i}.{docs,freqs,sizes}` files into a single `inv.{docs,freqs,sizes}`.

  pisa::invert::merge_batches (wrapped by _pisathon.merge_inv) reads exactly `final_term_count`
  posting lists from *every* batch file. A batch written before all terms were known (i.e. every
  batch except the last, when the term lexicon grows across batches) is physically shorter than
  that, so each such batch is padded here with explicit empty (length-0) posting lists for the
  terms introduced afterwards, before merging.
  """
  zero = np.zeros(1, dtype=np.uint32).tobytes()
  for bidx, term_count in batch_term_counts.items():
    pad = final_term_count - term_count
    if pad > 0:
      with (path/f'inv.batch.{bidx}.docs').open('ab') as f_did, (path/f'inv.batch.{bidx}.freqs').open('ab') as f_score:
        f_did.write(zero * pad)
        f_score.write(zero * pad)
  batch_count = max(batch_term_counts) + 1
  _pisathon.merge_inv(str(path/'inv'), batch_count, final_term_count)
  for i in range(batch_count):
    (path/f'inv.batch.{i}.docs').unlink()
    (path/f'inv.batch.{i}.freqs').unlink()
    (path/f'inv.batch.{i}.sizes').unlink()


def _remap_terms_alphabetically(path, lexicon):
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
      offsets_lens.append((int(i), int(in_docs[i]) + 1))
      i += int(in_docs[i]) + 1
    for term in _logger.pbar(sorted(lexicon), desc='re-mapping term ids'):
      f_lex.write(f'{term}\n')
      i = lexicon[term]
      start, doclen = offsets_lens[i]
      f_docs.write(in_docs[start:start+doclen])
      f_freqs.write(in_freqs[start:start+doclen])
  del in_docs # close mmap
  del in_freqs # close mmap
  (path/'inv.docs.tmp').unlink()
  (path/'inv.freqs.tmp').unlink()
  _pisathon.build_binlex(str(path/'fwd.documents'), str(path/'fwd.doclex'))
  _pisathon.build_binlex(str(path/'fwd.terms'), str(path/'fwd.termlex'))


def _finalize_append(path):
  """After an append, drops on-disk structures that are cached/lazily-built and would otherwise
  go stale: the compressed index + WAND data (rebuilt on first retrieval after this, see
  PisaRetrieve.reset_retrieval_context / _pisathon.prepare_index), and the raw `fwd` forward-index
  binary collection used by PisaIndex.get_corpus_iter, which append does not maintain."""
  for f in Path(path).iterdir():
    if '.bmw.' in f.name:
      f.unlink()
  fwd = Path(path)/'fwd'
  if fwd.exists():
    fwd.unlink()
