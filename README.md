# PyTerrier PISA

[PyTerrier](https://github.com/terrier-org/pyterrier) bindings for the [PISA](https://github.com/pisa-engine/pisa) search engine.

Interactive Colab Demo: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrierteam/pyterrier_pisa/blob/master/examples/pyterrier_pisa_antique.ipynb)

## Getting Started

These bindings are only available for cpython 3.7-3.10 on `manylinux2010_x86_64` platforms. They can be installed via pip:

```bash
pip install pyterrier_pisa
```

## Indexing

You can easily index corpora from PyTerrier datasets:

```python
import pyterrier as pt
if not pt.started():
  pt.init()
from pyterrier_pisa import PisaIndex

# from a dataset
dataset = pt.get_dataset('irds:msmarco-passage')
index = PisaIndex('./msmarco-passage-pisa')
index.index(dataset.get_corpus_iter())
```

You can also select which text field(s) to index. If not specified, all fields of type `str` will be indexed.

```python
dataset = pt.get_dataset('irds:cord19')
index = PisaIndex('./cord19-pisa', text_field=['title', 'abstract'])
index.index(dataset.get_corpus_iter())
```

`PisaIndex` accepts various other options to configure the indexing process. Most notable are:
 - `stemmer`: Which stemmer to use? Options: `porter2` (default), `krovetz`, `none`
 - `threads`: How many threads to use for indexing? Default: `8`
 - `index_encoding`: Which index encoding to use. Default: `block_simdbp`
 - `stops`: Which set of stopwords to use. Default: `terrier`.


```python
# E.g.,
index = PisaIndex('./cord19-pisa', stemmer='krovetz', threads=32)
```

For some collections you can download pre-built indices from [data.terrier.org](http://data.terrier.org/). PISA indices are prefixed with `pisa_`.

```python
index = PisaIndex.from_dataset('trec-covid')
```

## Retrieval

From an index, you can build retrieval transformers:

```python
dph = index.dph()
bm25 = index.bm25(k1=1.2, b=0.4)
pl2 = index.pl2(c=1.0)
qld = index.qld(mu=1000.)
```

These retrievers support all the typical pipeline operations.

Search:

```python
bm25.search('covid symptoms')
#     qid           query     docno     score
# 0     1  covid symptoms  a6avr09j  6.273450
# 1     1  covid symptoms  hdxs9dgu  6.272374
# 2     1  covid symptoms  zxq7dl9t  6.272374
# ..   ..             ...       ...       ...
# 999   1  covid symptoms  m8wggdc7  4.690651
```

Batch retrieval:

```python
print(dph(dataset.get_topics('title')))
#       qid                     query     docno     score
# 0       1        coronavirus origin  8ccl9aui  9.329109
# 1       1        coronavirus origin  es7q6c90  9.260190
# 2       1        coronavirus origin  8l411r1w  8.862670
# ...    ..                       ...       ...       ...
# 49999  50  mrna vaccine coronavirus  eyitkr3s  5.610429
```

Experiment:

```python
from pyterrier.measures import *
pt.Experiment(
  [dph, bm25, pl2, qld],
  dataset.get_topics('title'),
  dataset.get_qrels(),
  [nDCG@10, P@5, P(rel=2)@5, 'mrt'],
  names=['dph', 'bm25', 'pl2', 'qld']
)
#    name   nDCG@10    P@5  P(rel=2)@5       mrt
# 0   dph  0.623450  0.720       0.548  1.101846
# 1  bm25  0.624923  0.728       0.572  0.880318
# 2   pl2  0.536506  0.632       0.456  1.123883
# 3   qld  0.570032  0.676       0.504  0.974924
```

You can also build a retrieval transformer from `PisaRetrieve`:

```python
from pyterrier_pisa import PisaRetrieve
# from index path:
bm25 = PisaRetrieve('./cord19-pisa', scorer='bm25', bm25_k1=1.2, bm25_b=0.4)
# from dataset
bm25 = PisaRetrieve.from_dataset('trec-covid', 'pisa_unstemmed', scorer='bm25', bm25_k1=1.2, bm25_b=0.4)
```

## FAQ

**What retrieval functions are supported?**

 - `"dph"`. Parameters: (none)
 - `"bm25"`. Parameters: `k1`, `b`
 - `"pl2"`. Parameters: `c`
 - `"qld"`. Parameters: `mu`

**How do I index [some other type of data]?**

`PisaIndex` accepts an iterator over dicts, each of which containing a `docno` field and a `text` field. All you need to do is coerce the data into that
format and you're set.

Examples:

```python
# any iterator
def iter_docs():
  for i in range(100):
    yield {'docno': str(i), 'text': f'document {i}'}
index = PisaIndex('./dummy-pisa')
index.index(iter_docs())

# from a dataframe
import pandas as pd
docs = pd.DataFrame([
  ('1', 'test doc'),
  ('2', 'another doc'),
], columns=['docno', 'text'])
index = PisaIndex('./dummy-pisa-2')
index.index(docs.to_dict(orient="records"))
```

**Can I build a doc2query index?**

You can use `PisaIndex` with any document rewriter, such as doc2query or DeepCT.
All you need to do is build an indexing pipeline. For example:

```bash
pip install --upgrade git+https://github.com/terrierteam/pyterrier_doc2query.git
wget https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/t5-base.zip
unzip t5-base.zip
```

```python
doc2query = Doc2Query(out_attr="exp_terms", batch_size=8)
dataset = pt.get_dataset('irds:vaswani')
index = PisaIndex('./vaswani-doc2query-pisa')
index_pipeline = doc2query >> pt.apply.text(lambda r: f'{r["text"]} {r["exp_terms"]}') >> index
index_pipeline.index(dataset.get_corpus_iter())
```

**Can I build a learned sparse retrieval (e.g., SPLADE) index?**

Yes! Example:

```python
import pyt_splade
splade = pyt_splade.Splade()
dataset = pt.get_dataset('irds:msmarco-passage')
index = PisaIndex('./msmarco-passage-splade', stemmer='none')

# indexing
idx_pipeline = splade.doc_encoder() >> index.toks_indexer()
idx_pipeline.index(dataset.get_corpus_iter())

# retrieval

retr_pipeline = splade.query_encoder() >> index.quantized()
```

`msmarco-passage/trec-dl-2019` effectiveness for `naver/splade-cocondenser-ensembledistil`:

| System | nDCG@10 | R(rel=2)@1000 |
|--------|---------|---------------|
| PISA   | 0.731   |         0.872 |
| [From Paper](https://arxiv.org/pdf/2205.04733.pdf) | 0.732 | 0.875 |


**What are the supported index encodings and query algorithms?**

Right now we support the following index encodings: `ef`, `single`, `pefuniform`, `pefopt`, `block_optpfor`, `block_varintg8iu`, `block_streamvbyte`, `block_maskedvbyte`, `block_interpolative`, `block_qmx`, `block_varintgb`, `block_simple8b`, `block_simple16`, `block_simdbp`.

Index encodings are supplied when a `PisaIndex` is constructed:

```python
index = PisaIndex('./cord19-pisa', index_encoding='ef')
```

We support the following query algorithms: `wand`, `block_max_wand`, `block_max_maxscore`, `block_max_ranked_and`, `ranked_and`, `ranked_or`, `maxscore`.

Query algorithms are supplied when you construct a retrieval transformer:

```python
index.bm25(query_algorithm='ranked_and')
```

**Can I import/export from [CIFF](https://github.com/osirrc/ciff)?**

Yes! Using `.from_ciff(ciff_file, index_path)` and `.to_ciff(ciff_file)`

```python
# from a CIFF export:
index = PisaIndex.from_ciff('path/to/something.ciff', 'path/to/index.pisa', stemmer='krovetz') # stemmer is optional
# to a CIFF export:
index.to_ciff('path/to/something.ciff')
```

Note that you need to be careful to set stemmer to match whatever was used when constructing the index; CIFF does not directly store which stemmer
was used when building the index. If it's a stemmer that's not supported by PISA, you can set `stemmer='none'` and apply stemming in a PyTerrier pipeline.

## References

 - [Mallia19]: Antonio Mallia, Michal Siedlaczek, Joel Mackenzie, Torsten Suel. PISA: Performant Indexes and Search for Academia. Proceedings of the Open-Source IR Replicability Challenge. http://ceur-ws.org/Vol-2409/docker08.pdf
 - [MacAvaney22]: Sean MacAvaney, Craig Macdonald. A Python Interface to PISA!. Proceedings of SIGIR 2022.
 - [Macdonald21]: Craig Macdonald, Nicola Tonellotto, Sean MacAvaney, Iadh Ounis. PyTerrier: Declarative Experimentation in Python from BM25 to Dense Retrieval. Proceedings of CIKM 2021. https://dl.acm.org/doi/abs/10.1145/3459637.3482013

## Credits

 - Sean MacAvaney, University of Glasgow
 - Craig Macdonald, University of Glasgow
