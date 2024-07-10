// Python includes
#include <Python.h>


#include <boost/filesystem.hpp>



// STD includes
#include <stdio.h>

#include <forward_index_builder.hpp>
#include <parser.hpp>
#include <query/term_processor.hpp>
#include <gsl/span>
#include <algorithm>
#include <spdlog/spdlog.h>
#include <tbb/global_control.h>
#include <tbb/task_group.h>
#include <tbb/spin_mutex.h>

//#include <app.hpp>
#include <binary_collection.hpp>
#include <util/util.hpp>
#include <boost/algorithm/string.hpp>
#include <compress.hpp>
#include <fmt/format.h>
#include <forward_index_builder.hpp>
#include <invert.hpp>
#include <scorer/scorer.hpp>
#include <query/term_processor.hpp>
#include <wand_data.hpp>
#include <wand_utils.hpp>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <functional>
#include <mappable/mapper.hpp>
#include <mio/mmap.hpp>
#include <range/v3/view/enumerate.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include <accumulator/lazy_accumulator.hpp>
#include <cursor/block_max_scored_cursor.hpp>
#include <cursor/max_scored_cursor.hpp>
#include <cursor/scored_cursor.hpp>
#include <index_types.hpp>
#include <io.hpp>
#include <query/algorithm.hpp>
#include <scorer/scorer.hpp>
#include <tokenizer.hpp>
#include <type_alias.hpp>
#include <util/util.hpp>
#include <wand_data_compressed.hpp>
#include <wand_data_raw.hpp>


#include <reorder_docids.hpp>

#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

using pisa::Document_Record;
using pisa::Forward_Index_Builder;
using ranges::views::enumerate;
using namespace pisa;
namespace fs = boost::filesystem;


static PyTypeObject RetrievalContextType;

typedef struct {
    PyObject_HEAD
    void* index = NULL;
    std::shared_ptr<mio::mmap_source> docmap_source = NULL;
    std::shared_ptr<TermProcessor> term_processor = NULL;
    wand_data<wand_data_raw>* wdata = NULL;
} RetrievalContext;

static PyObject* RetrievalContext_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    return type->tp_alloc(type, 0);
}

static void RetrievalContext_dealloc(RetrievalContext* self) {
//   if (self->index != NULL) {
//     const char* encoding = self->encoding->c_str();
//     if (false) {  // NOLINT
// #define LOOP_BODY(R, DATA, T)                                                                      \
//     }                                                                                              \
//     else if (strcmp(encoding, BOOST_PP_STRINGIZE(T)) == 0)                                         \
//     {                                                                                              \
//       delete (BOOST_PP_CAT(T, _index)*)self->index;                                                \
//       /**/
//       BOOST_PP_SEQ_FOR_EACH(LOOP_BODY, _, PISA_INDEX_TYPES);
// #undef LOOP_BODY
//     }
//     self->index = NULL;
//   }

//   if (self->encoding != NULL) {
//     delete self->encoding;
//     self->encoding = NULL;
//   }

//   if (self->term_processor != NULL) {
//     delete self->term_processor;
//     self->term_processor = NULL;
//   }

//   // TODO: docmap? query_fun?

//   if (self->wdata != NULL) {
//     delete self->wdata;
//     self->wdata = NULL;
//   }
}

static PyObject *py_index(PyObject *self, PyObject *args) {
  const char* fin;
  const char* index_dir;
  const char* stemmer;
  int batch_size;
  int threads;

  /* Parse arguments */
  if(!PyArg_ParseTuple(args, "sssii", &fin, &index_dir, &stemmer, &batch_size, &threads)) {
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS

  std::ifstream ifs;
  ifs.open(fin);

  fs::path f_index_dir (index_dir);

  std::optional<std::string> stemmer_inp = std::nullopt;
  if (stemmer[0]) {
    stemmer_inp = stemmer;
  }
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads + 1);

  pisa::Forward_Index_Builder fwd_builder;
  fwd_builder.build(
        ifs,
        (f_index_dir/"fwd").string(),
        record_parser("plaintext", ifs),
        pisa::term_transformer_builder(stemmer_inp),
        pisa::parse_plaintext_content,
        batch_size,
        threads + 1);

  ifs.close();

  auto term_lexicon_file = (f_index_dir/"fwd.termlex").string();
  mio::mmap_source mfile(term_lexicon_file.c_str());
  auto lexicon = pisa::Payload_Vector<>::from(mfile);
  long unsigned int lex_size = lexicon.size();
  pisa::invert::InvertParams invert_params;
  invert_params.batch_size = batch_size;
  invert_params.num_threads = threads + 1;
  invert_params.term_count = lex_size;
  pisa::invert::invert_forward_index(
        (f_index_dir/"fwd").string(),
        (f_index_dir/"inv").string(),
        invert_params);

  Py_END_ALLOW_THREADS
  Py_RETURN_NONE;
}


static PyObject *py_merge_inv(PyObject *self, PyObject *args) {
  const char* fin;
  int batch_count;
  int term_count;

  /* Parse arguments */
  if(!PyArg_ParseTuple(args, "sii", &fin, &batch_count, &term_count)) {
      return NULL;
  }

  pisa::invert::merge_batches(fin, batch_count, term_count);

  Py_RETURN_NONE;
}


static PyObject *py_num_terms(PyObject *self, PyObject *args, PyObject *kwargs) {
  const char* index_dir;
  if(!PyArg_ParseTuple(args, "s", &index_dir)) {
      return NULL;
  }
  fs::path f_index_dir (index_dir);
  auto term_lexicon_file = (f_index_dir/"fwd.termlex").string();
  mio::mmap_source mfile(term_lexicon_file.c_str());
  auto lexicon = pisa::Payload_Vector<>::from(mfile);
  return PyLong_FromUnsignedLong(lexicon.size());
}


static PyObject *py_num_docs(PyObject *self, PyObject *args, PyObject *kwargs) {
  const char* index_dir;
  if(!PyArg_ParseTuple(args, "s", &index_dir)) {
      return NULL;
  }
  fs::path f_index_dir (index_dir);
  binary_freq_collection input_collection((f_index_dir/"inv").string().c_str());
  return PyLong_FromUnsignedLong(input_collection.num_docs());
}


static PyObject *py_prepare_index(PyObject *self, PyObject *args, PyObject *kwargs) {
  const char* index_dir;
  const char* scorer_name;
  const char* encoding;
  unsigned long long block_size = 64;
  unsigned int in_quantize = 0;
  float bm25_k1 = -100;
  float bm25_b = -100;
  float pl2_c = -100;
  float qld_mu = -100;
  unsigned int in_force = 0;

  /* Parse arguments */
  static const char *kwlist[] = {"index_dir", "encoding", "scorer_name", "block_size", "quantize", "bm25_k1", "bm25_b", "pl2_c", "qld_mu", "force", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sss|KIffffI", const_cast<char **>(kwlist),
                                     &index_dir, &encoding, &scorer_name, &block_size, &in_quantize, &bm25_k1, &bm25_b, &pl2_c, &qld_mu, &in_force))
  {
      return NULL;
  }

  bool quantize = in_quantize != 0;
  bool force = in_force != 0;
  auto scorer = ScorerParams(scorer_name);
  if (bm25_k1 != -100) scorer.bm25_k1 = bm25_k1;
  if (bm25_b  != -100) scorer.bm25_b  = bm25_b;
  if (pl2_c   != -100) scorer.pl2_c   = pl2_c;
  if (qld_mu  != -100) scorer.qld_mu  = qld_mu;

  std::string scorer_fmt;
       if (scorer.name == "bm25") scorer_fmt = fmt::format("{}.k1-{}.b-{}", scorer.name, scorer.bm25_k1, scorer.bm25_b);
  else if (scorer.name == "pl2") scorer_fmt = fmt::format("{}.c-{}", scorer.name, scorer.pl2_c);
  else if (scorer.name == "qld") scorer_fmt = fmt::format("{}.mu-{}", scorer.name, scorer.qld_mu);
  else if (scorer.name == "dph") scorer_fmt = scorer.name;
  else if (scorer.name == "quantized") scorer_fmt = scorer.name;
  else return NULL;

  fs::path f_index_dir (index_dir);
  fs::path wand_path = f_index_dir/fmt::format("{}.q{:d}.bmw.{:d}", scorer_fmt, quantize, block_size);
  fs::path comp_path (fmt::format("{}.{}", wand_path.string(), encoding));

  if (force || !fs::exists(wand_path)) {
    pisa::create_wand_data(
        wand_path.string(),
        (f_index_dir/"inv").string(),
        pisa::FixedBlock(block_size),
        scorer,
        false,
        false,
        quantize,
        {});
  }
  if (force || !fs::exists(comp_path)) {
    pisa::compress(
        (f_index_dir/"inv").string(),
        wand_path.string(),
        encoding,
        comp_path.string(),
        scorer,
        quantize,
        false);
  }

  Py_RETURN_NONE;
}




template <typename IndexType, typename WandType, typename ScorerFn>
static std::function<std::vector<typename topk_queue::entry_type>(Query)> get_query_processor(IndexType* index, WandType* wdata, const char* algorithm, unsigned int k, ScorerFn const& scorer, bool weighted) {
  std::function<std::vector<typename topk_queue::entry_type>(Query)> query_fun = NULL;

  if (strcmp(algorithm, "wand") == 0) {
    query_fun = [&, index, wdata, k, weighted](Query query) {
      topk_queue topk(k);
      wand_query wand_q(topk);
      wand_q(make_max_scored_cursors(*index, *wdata, *scorer, query, weighted), index->num_docs());
      topk.finalize();
      return topk.topk();
    };
  } else if (strcmp(algorithm, "block_max_wand") == 0) {
    query_fun = [&, index, wdata, k, weighted](Query query) {
      topk_queue topk(k);
      block_max_wand_query block_max_wand_q(topk);
      block_max_wand_q(
        make_block_max_scored_cursors(*index, *wdata, *scorer, query, weighted), index->num_docs());
      topk.finalize();
      return topk.topk();
    };
  } else if (strcmp(algorithm, "block_max_maxscore") == 0) {
    query_fun = [&, index, wdata, k, weighted](Query query) {
      topk_queue topk(k);
      block_max_maxscore_query block_max_maxscore_q(topk);
      block_max_maxscore_q(
        make_block_max_scored_cursors(*index, *wdata, *scorer, query, weighted), index->num_docs());
      topk.finalize();
      return topk.topk();
    };
  } else if (strcmp(algorithm, "block_max_ranked_and") == 0) {
    query_fun = [&, index, wdata, k, weighted](Query query) {
      topk_queue topk(k);
      block_max_ranked_and_query block_max_ranked_and_q(topk);
      block_max_ranked_and_q(
        make_block_max_scored_cursors(*index, *wdata, *scorer, query, weighted), index->num_docs());
      topk.finalize();
      return topk.topk();
    };
  } else if (strcmp(algorithm, "ranked_and") == 0) {
    query_fun = [&, index, wdata, k, weighted](Query query) {
      topk_queue topk(k);
      ranked_and_query ranked_and_q(topk);
      ranked_and_q(make_scored_cursors(*index, *scorer, query, weighted), index->num_docs());
      topk.finalize();
      return topk.topk();
    };
  } else if (strcmp(algorithm, "ranked_or") == 0) {
    query_fun = [&, index, wdata, k, weighted](Query query) {
      topk_queue topk(k);
      ranked_or_query ranked_or_q(topk);
      ranked_or_q(make_scored_cursors(*index, *scorer, query, weighted), index->num_docs());
      topk.finalize();
      return topk.topk();
    };
  } else if (strcmp(algorithm, "maxscore") == 0) {
    query_fun = [&, index, wdata, k, weighted](Query query) {
      topk_queue topk(k);
      maxscore_query maxscore_q(topk);
      maxscore_q(make_max_scored_cursors(*index, *wdata, *scorer, query, weighted), index->num_docs());
      topk.finalize();
      return topk.topk();
    };
  } else if (strcmp(algorithm, "ranked_or_taat") == 0) {
    query_fun = [&, index, wdata, k, weighted, accumulator = Simple_Accumulator(index->num_docs())](Query query) mutable {
      topk_queue topk(k);
      ranked_or_taat_query ranked_or_taat_q(topk);
      ranked_or_taat_q(
        make_scored_cursors(*index, *scorer, query, weighted), index->num_docs(), accumulator);
      topk.finalize();
      return topk.topk();
    };
  } else if (strcmp(algorithm, "ranked_or_taat_lazy") == 0) {
    query_fun = [&, index, wdata, k, weighted, accumulator = Lazy_Accumulator<4>(index->num_docs())](Query query) mutable {
      topk_queue topk(k);
      ranked_or_taat_query ranked_or_taat_q(topk);
      ranked_or_taat_q(
        make_scored_cursors(*index, *scorer, query), index->num_docs(), accumulator);
      topk.finalize();
      return topk.topk();
    };
  } else {
    spdlog::error("Unsupported query type: {}", algorithm);
  }

  return query_fun;
}


static PyObject *py_build_binlex(PyObject *self, PyObject *args, PyObject *kwargs) {
  const char* term_file;
  const char* termlex_file;

  /* Parse arguments */
  if(!PyArg_ParseTuple(args, "ss", &term_file, &termlex_file)) {
      return NULL;
  }

  std::string s_term_file(term_file);
  std::string s_termlex_file(termlex_file);

  std::ifstream is(s_term_file);
  encode_payload_vector(std::istream_iterator<io::Line>(is), std::istream_iterator<io::Line>()).to_file(s_termlex_file);

  Py_RETURN_NONE;
}





static PyObject *py_retrieve(PyObject *self, PyObject *args, PyObject *kwargs) {
  const char* index_dir;
  const char* encoding;
  const char* algorithm;
  const char* stemmer;
  const char* scorer_name;
  const char* stop_fname = "";
  int pretoks = 0;
  PyObject* in_queries;
  unsigned long long block_size = 64;
  unsigned int in_quantize = 0;
  unsigned int in_weighted = 0;
  unsigned int k = 1000;
  float bm25_k1 = -100;
  float bm25_b = -100;
  float pl2_c = -100;
  float qld_mu = -100;
  unsigned int threads = 8;

  Py_buffer result_qidxs;
  Py_buffer result_docnos;
  Py_buffer result_ranks;
  Py_buffer result_scores;

  /* Parse arguments */
  // Refer to the documentation for the kwarg type (character) definitions: https://docs.python.org/3/c-api/arg.html
  // Most notably: s: string, O: PyObject, K: unsigned long long, I: unsigned int, f: float, w*: Py_buffer
  static const char *kwlist[] = {"index_dir", "encoding", "algorithm", "scorer_name", "stemmer", "queries", "block_size", "quantize", "bm25_k1", "bm25_b", "pl2_c", "qld_mu", "k", "stop_fname", "threads", "pretokenised", "query_weighted", "result_qidxs", "result_docnos", "result_ranks", "result_scores", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssO|KIffffIsIiiw*w*w*w*", const_cast<char **>(kwlist),
                                     &index_dir, &encoding, &algorithm, &scorer_name, &stemmer, &in_queries, &block_size, &in_quantize, &bm25_k1, &bm25_b, &pl2_c, &qld_mu, &k, &stop_fname, &threads, &pretoks, &in_weighted, &result_qidxs, &result_docnos, &result_ranks, &result_scores))
  {
      return NULL;
  }

  auto in_queries_len = PyObject_Length(in_queries);
  if (in_queries_len == -1) {
    PyErr_SetString(PyExc_TypeError, "in_queries must provide __len__");
    return NULL;
  }

  fs::path f_index_dir (index_dir);

  std::optional<std::string> stemmer_inp = std::nullopt;
  if (stemmer[0]) {
    stemmer_inp = stemmer;
  }

  std::optional<std::string> stop_inp = std::nullopt;
  if (stop_fname[0]) {
    stop_inp = stop_fname;
  }

  auto term_processor = TermProcessor((f_index_dir/"fwd.termlex").string(), stop_inp, stemmer_inp);

  bool quantize = in_quantize != 0;
  auto scorer = ScorerParams(scorer_name);
  if (bm25_k1 != -100) scorer.bm25_k1 = bm25_k1;
  if (bm25_b  != -100) scorer.bm25_b  = bm25_b;
  if (pl2_c   != -100) scorer.pl2_c   = pl2_c;
  if (qld_mu  != -100) scorer.qld_mu  = qld_mu;

  std::string scorer_fmt;
       if (scorer.name == "bm25") scorer_fmt = fmt::format("{}.k1-{}.b-{}", scorer.name, scorer.bm25_k1, scorer.bm25_b);
  else if (scorer.name == "pl2") scorer_fmt = fmt::format("{}.c-{}", scorer.name, scorer.pl2_c);
  else if (scorer.name == "qld") scorer_fmt = fmt::format("{}.mu-{}", scorer.name, scorer.qld_mu);
  else if (scorer.name == "dph") scorer_fmt = scorer.name;
  else if (scorer.name == "quantized") scorer_fmt = scorer.name;
  else return NULL;

  auto wand_path = f_index_dir/fmt::format("{}.q{:d}.bmw.{:d}", scorer_fmt, quantize, block_size);
  auto index_path = (fmt::format("{}.{}", wand_path.string(), encoding));

  auto documents_path = f_index_dir/"fwd.doclex";

  std::function<std::vector<typename topk_queue::entry_type>(Query)> query_fun = NULL;
  wand_data<wand_data_raw>* wdata = new wand_data<wand_data_raw>(MemorySource::mapped_file(wand_path.string()));
  auto scorerf = scorer::from_params(scorer, *wdata);
  bool weighted = in_weighted == 1;

  void* index = NULL;

  /**/
  if (false) {  // NOLINT
#define LOOP_BODY(R, DATA, T)                                                                    \
  }                                                                                              \
  else if (strcmp(encoding, BOOST_PP_STRINGIZE(T)) == 0)                                         \
  {                                                                                              \
    index = new BOOST_PP_CAT(T, _index)(MemorySource::mapped_file(index_path));                  \
    query_fun = get_query_processor<BOOST_PP_CAT(T, _index)>((BOOST_PP_CAT(T, _index)*)index, wdata, algorithm, k, scorerf, weighted); \
    /**/

    BOOST_PP_SEQ_FOR_EACH(LOOP_BODY, _, PISA_INDEX_TYPES);
#undef LOOP_BODY
  } else {
    spdlog::error("Unknown type {}", encoding);
  }

  auto source = std::make_shared<mio::mmap_source>(documents_path.string().c_str());
  auto docmap = Payload_Vector<>::from(*source);
  int32_t *qidxs = (int32_t*)result_qidxs.buf;
  PyObject **docnos = (PyObject**)result_docnos.buf;
  int32_t *ranks = (int32_t*)result_ranks.buf;
  float_t *scores = (float_t*)result_scores.buf;

  auto iter = PyObject_GetIter(in_queries);
  tbb::spin_mutex mutex;
  size_t arr_idx = 0;

  tbb::parallel_for(size_t(0), size_t(threads), [&, query_fun](size_t thread_idx) {
        PyObject* res;
        int qidx;
        const char* qtext;
        mutex.lock();
        auto docnos_tmp = new std::string[k];
        while (1) {
          if (PyErr_CheckSignals() != 0) {
            break;
          }
          res = PyIter_Next(iter);
          if (res == NULL) {
            break;
          }
          Query query;
          std::vector<term_id_type> parsed_query;
          if (pretoks) {
            PyObject* qtermsdict;
            // tuple of string and dictiorary, where each entry contains a term and float weight
            PyArg_ParseTuple(res, "iO", &qidx, &qtermsdict);
            PyObject *termKey, *weightValue;
            Py_ssize_t pos = 0;
            while (PyDict_Next(qtermsdict, &pos, &termKey, &weightValue)) {
              // term
              const char* term_string = PyUnicode_AsUTF8(termKey);
              if (term_string == NULL && PyErr_Occurred()) {
                PyErr_SetString(PyExc_TypeError, "token string could not be parsed");
                break;
              }
              //we assume that stemming and stopwords are disabled here
              //and hence term_processor is a basic one.
              auto term = term_processor(term_string);
              if (term) {
                // weight
                double weight = PyFloat_AS_DOUBLE(weightValue);
                if (weight == -1.0 && PyErr_Occurred()) {
                  PyErr_SetString(PyExc_TypeError, "tok weights must be double");
                  break;
                }
                // Doesn't look like PISA uses the query_weights for anything; instead, we gotta repeat the query terms
                for (int i=1; i<=weight; i++) {
                  parsed_query.push_back(*term);
                }
              }
            }
            query = {"", std::move(parsed_query), {}};

          } else {
            PyArg_ParseTuple(res, "is", &qidx, &qtext);
            auto tokenstream = EnglishTokenizer().tokenize(qtext);
            std::vector<term_id_type> parsed_query;
            for (auto term_iter = tokenstream->begin(); term_iter != tokenstream->end(); ++term_iter) {
              auto raw_term = *term_iter;
              auto term = term_processor(raw_term);
              if (term && !term_processor.is_stopword(*term)) parsed_query.push_back(*term);
            }
            query = {"", std::move(parsed_query), {}};
          }
          Py_DECREF(res);

          mutex.unlock();
          auto query_res = query_fun(query);
          // Stabilise the sort by sorting on score (desc), then docid (asc). See <https://github.com/pisa-engine/pisa/issues/508>
          std::sort(query_res.begin(), query_res.end(), [](auto a, auto b) {
            return a.first == b.first ? a.second < b.second : a.first > b.first;
          });
          mutex.lock();
          auto count = query_res.size();
          size_t start = arr_idx;
          size_t i = 0;
          arr_idx += count;
          mutex.unlock();
          for (auto r: query_res) {
            docnos_tmp[i] = docmap[r.second];
            qidxs[start+i] = qidx;
            ranks[start+i] = i;
            scores[start+i] = r.first;
            i++;
          }
          mutex.lock();
          for (int i=0; i<count; ++i) {
            auto docno = PyUnicode_FromStringAndSize(docnos_tmp[i].data(), docnos_tmp[i].length());
            docnos[start+i] = docno; // takes ownership, shouldn't decref
          }
        }
      mutex.unlock();
      delete [] docnos_tmp;
  });

  Py_DECREF(iter);
  if (PyErr_CheckSignals() != 0) {
    return NULL;
  }

  if (false) {  // NOLINT
#define LOOP_BODY(R, DATA, T)                                                                    \
  }                                                                                              \
  else if (strcmp(encoding, BOOST_PP_STRINGIZE(T)) == 0)                                         \
  {                                                                                              \
    delete (BOOST_PP_CAT(T, _index)*)index;                                                      \
    /**/

    BOOST_PP_SEQ_FOR_EACH(LOOP_BODY, _, PISA_INDEX_TYPES);
#undef LOOP_BODY
  }
  delete wdata;

  PyBuffer_Release(&result_qidxs);
  PyBuffer_Release(&result_docnos);
  PyBuffer_Release(&result_ranks);
  PyBuffer_Release(&result_scores);
  PyObject *result = PyLong_FromLong(arr_idx);

  if (PyErr_CheckSignals() != 0) {
    return NULL;
  }

  return result;
}


static PyObject *py_prepare_index3(PyObject *self, PyObject *args, PyObject *kwargs) {
  RetrievalContext* ctxt;
  auto start = std::chrono::high_resolution_clock::now();
  const char* index_dir;
  const char* encoding;
  const char* algorithm;
  const char* stemmer;
  const char* scorer_name;
  const char* stop_fname = "";
  int pretoks = 0;
  unsigned long long block_size = 64;
  unsigned int in_quantize = 0;
  unsigned int in_weighted = 0;
  unsigned int k = 1000;
  float bm25_k1 = -100;
  float bm25_b = -100;
  float pl2_c = -100;
  float qld_mu = -100;
  unsigned int threads = 1;

  /* Parse arguments */
  // Refer to the documentation for the kwarg type (character) definitions: https://docs.python.org/3/c-api/arg.html
  // Most notably: s: string, O: PyObject, K: unsigned long long, I: unsigned int, f: float, w*: Py_buffer
  static const char *kwlist[] = {"context", "index_dir", "encoding", "algorithm", "scorer_name", "stemmer", "block_size", "quantize", "bm25_k1", "bm25_b", "pl2_c", "qld_mu", "stop_fname", "query_weighted", "k", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Osssss|KIffffsiI", const_cast<char **>(kwlist),
                                     &ctxt, &index_dir, &encoding, &algorithm, &scorer_name, &stemmer, &block_size, &in_quantize, &bm25_k1, &bm25_b, &pl2_c, &qld_mu, &stop_fname, &in_weighted, &k))
  {
      return NULL;
  }

  fs::path f_index_dir (index_dir);

  std::optional<std::string> stemmer_inp = std::nullopt;
  if (stemmer[0]) {
    stemmer_inp = stemmer;
  }

  std::optional<std::string> stop_inp = std::nullopt;
  if (stop_fname[0]) {
    stop_inp = stop_fname;
  }

  ctxt->term_processor = std::make_shared<TermProcessor>(TermProcessor((f_index_dir/"fwd.termlex").string(), stop_inp, stemmer_inp));

  bool quantize = in_quantize != 0;
  auto scorer = ScorerParams(scorer_name);
  if (bm25_k1 != -100) scorer.bm25_k1 = bm25_k1;
  if (bm25_b  != -100) scorer.bm25_b  = bm25_b;
  if (pl2_c   != -100) scorer.pl2_c   = pl2_c;
  if (qld_mu  != -100) scorer.qld_mu  = qld_mu;

  std::string scorer_fmt;
       if (scorer.name == "bm25") scorer_fmt = fmt::format("{}.k1-{}.b-{}", scorer.name, scorer.bm25_k1, scorer.bm25_b);
  else if (scorer.name == "pl2") scorer_fmt = fmt::format("{}.c-{}", scorer.name, scorer.pl2_c);
  else if (scorer.name == "qld") scorer_fmt = fmt::format("{}.mu-{}", scorer.name, scorer.qld_mu);
  else if (scorer.name == "dph") scorer_fmt = scorer.name;
  else if (scorer.name == "quantized") scorer_fmt = scorer.name;
  else return NULL;

  auto wand_path = f_index_dir/fmt::format("{}.q{:d}.bmw.{:d}", scorer_fmt, quantize, block_size);
  auto index_path = (fmt::format("{}.{}", wand_path.string(), encoding));

  auto documents_path = f_index_dir/"fwd.doclex";

  // auto wdata_source = std::make_shared<mio::mmap_source>(wand_path.string());
  // ctxt->wdata_source = std::make_shared<mio::mmap_source>(wand_path.string());
  auto wdata = new wand_data<wand_data_raw>(MemorySource::mapped_file(wand_path.string()));
  ctxt->wdata = wdata;

  auto scorerf = scorer::from_params(scorer, *wdata);
  bool weighted = in_weighted == 1;

  /**/
  if (false) {  // NOLINT
#define LOOP_BODY(R, DATA, T)                                                                    \
  }                                                                                              \
  else if (strcmp(encoding, BOOST_PP_STRINGIZE(T)) == 0)                                         \
  {                                                                                              \
    ctxt->index = new BOOST_PP_CAT(T, _index)(MemorySource::mapped_file(index_path));            \
  /**/
    BOOST_PP_SEQ_FOR_EACH(LOOP_BODY, _, PISA_INDEX_TYPES);
#undef LOOP_BODY
  } else {
    spdlog::error("(prepare_index) Unknown type {}", encoding);
  }

  ctxt->docmap_source = std::make_shared<mio::mmap_source>(documents_path.string().c_str());

  Py_RETURN_NONE;
}


static PyObject *py_retrieve4(PyObject *self, PyObject *args, PyObject *kwargs) {
  RetrievalContext* ctxt;
  const char* index_dir;
  const char* encoding;
  const char* algorithm;
  const char* stemmer;
  const char* scorer_name;
  const char* stop_fname = "";
  int pretoks = 0;
  PyObject* in_queries;
  unsigned long long block_size = 64;
  unsigned int in_quantize = 0;
  unsigned int in_weighted = 0;
  unsigned int k = 1000;
  float bm25_k1 = -100;
  float bm25_b = -100;
  float pl2_c = -100;
  float qld_mu = -100;
  unsigned int threads = 1;

  Py_buffer result_qidxs;
  Py_buffer result_docnos;
  Py_buffer result_ranks;
  Py_buffer result_scores;

  /* Parse arguments */
  // Refer to the documentation for the kwarg type (character) definitions: https://docs.python.org/3/c-api/arg.html
  // Most notably: s: string, O: PyObject, K: unsigned long long, I: unsigned int, f: float, w*: Py_buffer
  static const char *kwlist[] = {"context", "index_dir", "encoding", "algorithm", "scorer_name", "stemmer", "queries", "block_size", "quantize", "bm25_k1", "bm25_b", "pl2_c", "qld_mu", "k", "stop_fname", "threads", "pretokenised", "query_weighted", "result_qidxs", "result_docnos", "result_ranks", "result_scores", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OsssssO|KIffffIsIiiw*w*w*w*", const_cast<char **>(kwlist),
                                     &ctxt, &index_dir, &encoding, &algorithm, &scorer_name, &stemmer, &in_queries, &block_size, &in_quantize, &bm25_k1, &bm25_b, &pl2_c, &qld_mu, &k, &stop_fname, &threads, &pretoks, &in_weighted, &result_qidxs, &result_docnos, &result_ranks, &result_scores))
  {
      return NULL;
  }


  auto in_queries_len = PyObject_Length(in_queries);
  if (in_queries_len == -1) {
    PyErr_SetString(PyExc_TypeError, "in_queries must provide __len__");
    return NULL;
  }

  fs::path f_index_dir (index_dir);

  TermProcessor term_processor = *(ctxt->term_processor);

  bool quantize = in_quantize != 0;
  auto scorer = ScorerParams(scorer_name);
  if (bm25_k1 != -100) scorer.bm25_k1 = bm25_k1;
  if (bm25_b  != -100) scorer.bm25_b  = bm25_b;
  if (pl2_c   != -100) scorer.pl2_c   = pl2_c;
  if (qld_mu  != -100) scorer.qld_mu  = qld_mu;

  std::string scorer_fmt;
       if (scorer.name == "bm25") scorer_fmt = fmt::format("{}.k1-{}.b-{}", scorer.name, scorer.bm25_k1, scorer.bm25_b);
  else if (scorer.name == "pl2") scorer_fmt = fmt::format("{}.c-{}", scorer.name, scorer.pl2_c);
  else if (scorer.name == "qld") scorer_fmt = fmt::format("{}.mu-{}", scorer.name, scorer.qld_mu);
  else if (scorer.name == "dph") scorer_fmt = scorer.name;
  else if (scorer.name == "quantized") scorer_fmt = scorer.name;
  else return NULL;

  auto scorerf = scorer::from_params(scorer, *ctxt->wdata);
  bool weighted = in_weighted == 1;

  std::function<std::vector<typename topk_queue::entry_type>(Query)> query_fun = NULL;
  if (false) {  // NOLINT
#define LOOP_BODY(R, DATA, T)                                                                    \
  }                                                                                              \
  else if (strcmp(encoding, BOOST_PP_STRINGIZE(T)) == 0)                                         \
  {                                                                                              \
    query_fun = get_query_processor<BOOST_PP_CAT(T, _index)>((BOOST_PP_CAT(T, _index)*)ctxt->index, ctxt->wdata, algorithm, k, scorerf, weighted); \
    /**/
    BOOST_PP_SEQ_FOR_EACH(LOOP_BODY, _, PISA_INDEX_TYPES);
#undef LOOP_BODY
  } else {
    spdlog::error("(retrieve) Unknown type {}", encoding);
  }
    /**/

  auto docmap = Payload_Vector<>::from(*ctxt->docmap_source);

  auto qidxs  = (int32_t*)  result_qidxs.buf;
  auto docnos = (PyObject**)result_docnos.buf;
  auto ranks  = (int32_t*)  result_ranks.buf;
  auto scores = (float_t*)  result_scores.buf;

  auto iter = PyObject_GetIter(in_queries);
  tbb::spin_mutex mutex;
  size_t arr_idx = 0;

  auto run = [&, threads](size_t thread_idx) {
    PyObject* res;
    int qidx;
    const char* qtext;
    if (threads != 1) mutex.lock();
    auto docnos_tmp = new std::string[k];
    std::chrono::time_point<std::chrono::high_resolution_clock> query_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> query_end;
    while (1) {
      if (PyErr_CheckSignals() != 0) {
        break;
      }
      res = PyIter_Next(iter);
      if (res == NULL) {
        break;
      }
      Query query;
      std::vector<term_id_type> parsed_query;
      if (pretoks) {
        PyObject* qtermsdict;
        // tuple of string and dictiorary, where each entry contains a term and float weight
        PyArg_ParseTuple(res, "iO", &qidx, &qtermsdict);
        PyObject *termKey, *weightValue;
        Py_ssize_t pos = 0;
        while (PyDict_Next(qtermsdict, &pos, &termKey, &weightValue)) {
          // term
          const char* term_string = PyUnicode_AsUTF8(termKey);
          if (term_string == NULL && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "token string could not be parsed");
            break;
          }
          //we assume that stemming and stopwords are disabled here
          //and hence term_processor is a basic one.
          auto term = term_processor(term_string);
          if (term) {
            // weight
            double weight = PyFloat_AS_DOUBLE(weightValue);
            if (weight == -1.0 && PyErr_Occurred()) {
              PyErr_SetString(PyExc_TypeError, "tok weights must be double");
              break;
            }
            // Doesn't look like PISA uses the query_weights for anything; instead, we gotta repeat the query terms
            for (int i=1; i<=weight; i++) {
              parsed_query.push_back(*term);
            }
          }
        }
        query = {"", std::move(parsed_query), {}};

      } else {
        PyArg_ParseTuple(res, "is", &qidx, &qtext);
        auto tokenstream = EnglishTokenizer().tokenize(qtext);
        std::vector<term_id_type> parsed_query;
        for (auto term_iter = tokenstream->begin(); term_iter != tokenstream->end(); ++term_iter) {
          auto raw_term = *term_iter;
          auto term = term_processor(raw_term);
          if (term && !term_processor.is_stopword(*term)) {
            parsed_query.push_back(*term);
          }
        }
        query = {"", std::move(parsed_query), {}};
      }
      Py_DECREF(res);

      if (threads != 1) mutex.unlock();

      auto query_res = query_fun(query);
      // Stabilise the sort by sorting on score (desc), then docid (asc). See <https://github.com/pisa-engine/pisa/issues/508>
      std::sort(query_res.begin(), query_res.end(), [](auto a, auto b) {
        return a.first == b.first ? a.second < b.second : a.first > b.first;
      });
      auto count = query_res.size();
      if (threads != 1) mutex.lock();
      size_t start = arr_idx;
      arr_idx += count;
      if (threads != 1) mutex.unlock();
      size_t i = 0;
      for (auto r: query_res) {
        docnos_tmp[i] = docmap[r.second];
        scores[start+i] = r.first;
        ranks[start+i] = i;
        i++;
      }
      std::fill(qidxs+start, qidxs+start+count, qidx);

      if (threads != 1) mutex.lock();
      for (int i=0; i<count; ++i) {
        auto docno = PyUnicode_FromStringAndSize(docnos_tmp[i].data(), docnos_tmp[i].length());
        docnos[start+i] = docno; // takes ownership, shouldn't decref
      }
    }
    if (threads != 1) mutex.unlock();
    delete [] docnos_tmp;
  };

  if (threads == 1) {
    run(0); // no need to start up extra threads
  }
  else {
    tbb::parallel_for(size_t(0), size_t(threads), run);
  }

  Py_DECREF(iter);
  if (PyErr_CheckSignals() != 0) {
    return NULL;
  }

  PyBuffer_Release(&result_qidxs);
  PyBuffer_Release(&result_docnos);
  PyBuffer_Release(&result_ranks);
  PyBuffer_Release(&result_scores);

  PyObject *result = PyLong_FromLong(arr_idx);

  if (PyErr_CheckSignals() != 0) {
    return NULL;
  }

  return result;
}



static PyObject *py_log_level(PyObject *self, PyObject *args, PyObject *kwargs) {
  int level;
  if(!PyArg_ParseTuple(args, "i", &level)) {
      return NULL;
  }
  if (level == 0) {
    spdlog::set_default_logger(spdlog::create<spdlog::sinks::null_sink_mt>("stderr"));
  } else {
    spdlog::set_default_logger(spdlog::stderr_color_mt("stderr"));
  }
  Py_RETURN_NONE;
}

static PyObject *py_tokenize(PyObject *self, PyObject *args, PyObject *kwargs) {
  const char* stemmer;
  const char* text;

  /* Parse arguments */
  if(!PyArg_ParseTuple(args, "ss", &text, &stemmer)) {
      return NULL;
  }

  std::optional<std::string> stemmer_inp = std::nullopt;
  if (stemmer[0]) {
    stemmer_inp = stemmer;
  }
  auto py_toks = PyList_New(0);
  auto stem = pisa::term_transformer_builder(stemmer_inp)();
  pisa::parse_plaintext_content(text, [py_toks, stem](std::string&& inp) {
    PyList_Append(py_toks, PyUnicode_FromString(stem(inp).c_str()));
  });

  return py_toks;
}


static PyMethodDef pisathon_methods[] = {
  {"index", py_index, METH_VARARGS, "index"},
  {"merge_inv", py_merge_inv, METH_VARARGS, "merge_inv"},
  {"prepare_index", (PyCFunction) py_prepare_index, METH_VARARGS | METH_KEYWORDS, "prepare_index"},
  {"prepare_index3", (PyCFunction) py_prepare_index3, METH_VARARGS | METH_KEYWORDS, "prepare_index3"},
  {"retrieve", (PyCFunction)py_retrieve, METH_VARARGS | METH_KEYWORDS, "retrieve"},
  {"retrieve4", (PyCFunction)py_retrieve4, METH_VARARGS | METH_KEYWORDS, "retrieve4"},
  {"num_terms", (PyCFunction)py_num_terms, METH_VARARGS, "num_terms"},
  {"num_docs", (PyCFunction)py_num_docs, METH_VARARGS, "num_docs"},
  {"log_level", (PyCFunction)py_log_level, METH_VARARGS, "log_level"},
  {"build_binlex", (PyCFunction)py_build_binlex, METH_VARARGS, "build_binlex"},
  {"tokenize", (PyCFunction)py_tokenize, METH_VARARGS, "tokenize"},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

//-----------------------------------------------------------------------------
#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init__pisathon(void)
{
  (void) Py_InitModule("_pisathon", pisathon_methods);
}
#else /* PY_MAJOR_VERSION >= 3 */
static struct PyModuleDef pisathon_module_def = {
  PyModuleDef_HEAD_INIT,
  "_pisathon",
  "Internal \"_pisathon\" module for pyterrier_pisa",
  -1,
  pisathon_methods
};

PyMODINIT_FUNC PyInit__pisathon(void)
{
  PyObject* const module = PyModule_Create(&pisathon_module_def);
  if (module == NULL) {
    return NULL;
  }

  PyTypeObject RetrievalContextType_local = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "pyterrier_pisa._pisathon.RetrievalContext",   /* tp_name */
        sizeof(RetrievalContext),         /* tp_basicsize */
        0,                         /* tp_itemsize */
        0, // (destructor) RetrievalContext_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        0,                         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
        "RetrievalContext object",       /* tp_doc */
        0,                         /* tp_traverse */
        0,                         /* tp_clear */
        0,                         /* tp_richcompare */
        0,                         /* tp_weaklistoffset */
        0,                         /* tp_iter */
        0,                         /* tp_iternext */
        0,                         /* tp_methods */
        0,                         /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        0,                         /* tp_init */
        0,                         /* tp_alloc */
        RetrievalContext_new,             /* tp_new */
  };
  RetrievalContextType = RetrievalContextType_local;
  if (PyType_Ready(&RetrievalContextType) < 0) {
    return NULL;
  }
  Py_INCREF(&RetrievalContextType);
  PyModule_AddObject(module, "RetrievalContext", (PyObject*) &RetrievalContextType);
  return module;
}
#endif /* PY_MAJOR_VERSION >= 3 */
