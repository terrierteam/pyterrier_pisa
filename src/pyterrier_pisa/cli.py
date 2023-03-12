import argparse
import sys
import pyterrier as pt
from pyterrier_pisa import PisaIndex, PisaRetrieve, PisaScorer, PisaStopwords, PISA_INDEX_DEFAULTS


def main():
  if not pt.started():
    pt.init()
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
