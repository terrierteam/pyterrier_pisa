PISA + PyTerrier
=================================================

`pyterrier-pisa <https://github.com/terrierteam/pyterrier_pisa>`__ provids PyTerrier bindings to the
`PISA engine <https://github.com/pisa-engine/pisa>`__. PISA provides very efficient sparse indexing and retrieval.

Getting Started
-------------------------------------------------

You can install ``pyterrier-pisa`` using pip:

.. code-block:: console
   :caption: Install ``pyterrier-pisa`` with ``pip``

   pip install pyterrier-pisa

.. attention::

   ``pyterrier-pisa`` is only available on linux (``manylinux2010_x86_64``) platforms at this time.
   There are pre-built images for Python 3.8-3.11 on pypi.

The main class is :class:`~pyterrier_pisa.PisaIndex`. It provides functionality for indexing and retrieval.

Indexing
------------------------------------------------

You can easily index corpora from PyTerrier datasets:

.. code-block:: python
   :caption: Index using PISA

   import pyterrier as pt
   from pyterrier_pisa import PisaIndex

   # from a dataset
   dataset = pt.get_dataset('irds:msmarco-passage')
   index = PisaIndex('./msmarco-passage-pisa')
   index.index(dataset.get_corpus_iter())

You can also select which text field(s) to index. If not specified, all fields of type `str` will be indexed.

.. code-block:: python
   :caption: Choosing the fields to index with PISA

   dataset = pt.get_dataset('irds:cord19')
   index = PisaIndex('./cord19-pisa', text_field=['title', 'abstract'])
   index.index(dataset.get_corpus_iter())

Retrieval
------------------------------------------------

From an index, you can build retrieval transformers:

.. code-block:: python
   :caption: Constructing PISA retrieval transformers

   dph = index.dph()
   bm25 = index.bm25(k1=1.2, b=0.4)
   pl2 = index.pl2(c=1.0)
   qld = index.qld(mu=1000.)

These retrievers support all the typical pipeline operations.

Search:

.. code-block:: python
   :caption: Searching with a PISA retriever

   bm25.search('covid symptoms')
   #     qid           query     docno     score
   # 0     1  covid symptoms  a6avr09j  6.273450
   # 1     1  covid symptoms  hdxs9dgu  6.272374
   # 2     1  covid symptoms  zxq7dl9t  6.272374
   # ..   ..             ...       ...       ...
   # 999   1  covid symptoms  m8wggdc7  4.690651

Batch retrieval:

.. code-block:: python
   :caption: Batch retrieval with a PISA retriever

   print(dph(dataset.get_topics('title')))
   #       qid                     query     docno     score
   # 0       1        coronavirus origin  8ccl9aui  9.329109
   # 1       1        coronavirus origin  es7q6c90  9.260190
   # 2       1        coronavirus origin  8l411r1w  8.862670
   # ...    ..                       ...       ...       ...
   # 49999  50  mrna vaccine coronavirus  eyitkr3s  5.610429

Experiment:

.. code-block:: python
   :caption: Conducting an experiment with PISA retrievers

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

Extras
------------------------------------------------

- You can upload/download indexes to/from HuggingFace Hub using :meth:`~pyterrier_pisa.PisaIndex.to_hf` and :meth:`~pyterrier_pisa.PisaIndex.from_hf`.
- You can access PISA's tokenizers and stemmers using the :func:`~pyterrier_pisa.tokenize`.

API Documentation
------------------------------------------------

.. autoclass:: pyterrier_pisa.PisaIndex
   :members:

   .. automethod:: to_hf

      Upload a PISA index to HuggingFace Hub

   .. automethod:: from_hf

      Load a PISA index from HuggingFace Hub

.. autoenum:: pyterrier_pisa.PisaStemmer
.. autoenum:: pyterrier_pisa.PisaScorer
.. autoenum:: pyterrier_pisa.PisaIndexEncoding
.. autoenum:: pyterrier_pisa.PisaQueryAlgorithm
.. autoenum:: pyterrier_pisa.PisaStopwords
.. autofunction:: pyterrier_pisa.tokenize

References
------------------------------------------------

.. cite.dblp:: conf/sigir/MalliaSMS19
.. cite.dblp:: conf/sigir/MacAvaneyM22
