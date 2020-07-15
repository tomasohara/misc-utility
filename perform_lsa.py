#! /usr/bin/env python
#
# perform_lsa.py: Performs latent semantic analysis (LSA) matrix decomposition via gensim.
#
# Notes:
# - based on http://radimrehurek.com/gensim/wiki.html and http://radimrehurek.com/gensim/tut3.html
# - model parameters:
#   - LsiModel(corpus=None, num_topics=200, id2word=None, chunksize=20000,
#              decay=1.0, distributed=False, onepass=True,
#              power_iters=P2_EXTRA_ITERS, extra_samples=P2_EXTRA_DIMS)
#   - LdaModel(corpus=None, num_topics=100, id2word=None, distributed=False,
#              chunksize=2000, passes=1, update_every=1, alpha='symmetric', eta=None, decay=0.5,
#              eval_every=10, iterations=50, gamma_threshold=0.001):
# - The support for remote LSA assumes tcsh installed.
#
# TODO:
# - Add option for automatically invoking preprocessing if document-term matrix unavailable. 
# - Have option to save results of similarity calculations.
# - Use python script for killing processes based on pattern (instead of kill_em.sh).
# - Address ymeng's review:
#   After L46, parameter list would be nice.
# - Use remote_dispatching.py to encapsulate the remote processing support.
#

"""Module for performing latent semantic analysis (LSA) matrix decomposition (via gensim)."""

import argparse
import logging
import numpy
import os
import sys

import gensim

import tpo_common as tpo
import glue_helpers as gh
from gensim_test import CorpusData, resolve_terms, SimilarDocumentByCosine, MAX_SIMILAR, DOCID_FILENAME

NUM_TOPICS = tpo.getenv_integer("NUM_TOPICS", 400, "Number of topics for matrix decomposition")
TOP_TOPICS = tpo.getenv_integer("TOP_TOPICS", NUM_TOPICS, "Number of top topics to display")
NUM_DOCS = tpo.getenv_integer("NUM_DOCS", -1, "Number of documents for dictionary fixup")
PYTHON = tpo.getenv_text("PYTHON", "python")
RANDOM_SEED = tpo.getenv_integer("RANDOM_SEED", -1,
                                 "Integral seed for random number generation")

#------------------------------------------------------------------------

class SimilarDocumentByLSA(SimilarDocumentByCosine):
    """Class for finding similar documents via latent semantic analysis"""

    def __init__(self, corpus=None, dictionary=None, lsa=None, index_file=None, verbose_output=False, max_similar=MAX_SIMILAR, docid_filename=DOCID_FILENAME):
        """Class constructor"""
        tpo.debug_format("SimilarDocumentByLSA.__init__({corpus}, {dictionary}, {lsa}. {index_file}, {verbose_output}, {max_similar}, {docid_filename})", 6)
        self.lsa = lsa
        SimilarDocumentByCosine.__init__(self, corpus, dictionary, index_file, verbose_output, max_similar, docid_filename)
        # TEMP: force shard prefix check
        if index_file and gh.non_empty_file(index_file):
            self.sim_index.output_prefix = index_file
            self.sim_index.check_moved()
        return

    def find(self, docid):
        """"Return documents similar to DOCID; result is a list of tuples: (docid, weight)"""
        tpo.debug_format("SimilarDocumentByLSA.find({docid})", 5)
        gh.assertion(self.corpus and self.dictionary and self.lsa)
        gensim_docid = self.get_gensim_id(docid)
        try:
            similar_gensim_docs = self.sim_index[self.lsa[self.corpus[int(gensim_docid)]]]
            similar_docs = [(self.get_user_id(doc), self.normalize_score(score)) for (doc, score) in similar_gensim_docs]
            if self.verbose_output:
                similar_docs = [(docid, score, resolve_terms(docid, self.dictionary)) for (docid, score) in similar_docs]
            # TODO: ensure docid's are strings
        except:
            tpo.debug_raise()
            tpo.print_stderr("Exception retrieving similar documents: " + str(sys.exc_info()))
            similar_docs = []
        result = similar_docs
        tpo.debug_format("find({docid}) => {result}", 5)
        return (similar_docs)

    def derive_all_similarities(self):
        """Precompute similarities, using batch method via chunking (see Gensim documentation in docsim.py)."""
        tpo.debug_format("SimilarDocumentByLSA.derive_all_similarities()", 5)
        _all_sim = list(self.sim_index[self.lsa[self.corpus]])
        return

#------------------------------------------------------------------------
# Support for remote script processing
# note: remote_dispatching.py now handles most if this (exept for pyro support)

def run_command_on_workers(workers, command):
    "Run COMMAND on each of WORKERS"
    juju_rsa_file = gh.resolve_path("juju-id_rsa")
    for h in workers:
        gh.run("ssh -i {rsa} {h} {cmd}", 4, 
               ra=juju_rsa_file, cmd=command, h=h)
    return


def copy_file_to_workers(workers, filename):
    "Uploads FILENAME to each of WORKERS under /tmp"
    juju_rsa_file = gh.resolve_path("juju-id_rsa")
    for h in workers:
        gh.run("scp -i {rsa} {file} root@{h}:/tmp", 4, 
               rsa=juju_rsa_file, file=filename, h=h)
    return


def setup_remote_workers(remote_workers):
    """Setup each of the REMOTE_WORKERS to act as clients for LSA processing"""
    # Copy the package setup script to remote hosts and run
    setup_script = "/tmp/setup_remote_workers.sh"
    gh.write_lines(setup_script, 
                   ["apt-get --yes install python-numpy python-scipy",
                    "easy_install Pyro4",
                    "easy_install gensim",
                    "apt-get install tcsh"])
    copy_file_to_workers(remote_workers, setup_script)
    run_command_on_workers(remote_workers, tpo.format("source {setup_script}"))

    # Copy kill_em.sh to remote hosts for convenient worker termination; also, put locally in /tmp.
    kill_script = gh.resolve_path("kill_em.sh")
    copy_file_to_workers(remote_workers, kill_script)
    gh.run("cp {kill_script} /tmp")
    return


def start_remote_workers(remote_workers):
    """Start lsi_worker interface on each of the REMOTE_WORKERS along with local dispatcher"""

    # Start local dispatcher
    PYRO_NS_HOST = tpo.getenv_text("PYRO_NS_HOST", gh.run("uname -n"))
    ## BAD: gh.run("PYRO_NS_HOST={PYRO_NS_HOST} PYRO_SERIALIZERS_ACCEPTED=pickle PYRO_SERIALIZER=pickle  {PYTHON} -m Pyro4.naming --port 9090 >| run-pyro-host-port.log 2>&1 &", just_issue=True)
    ## BAD: gh.run("PYRO_NS_HOST={PYRO_NS_HOST} {PYTHON} -m gensim.models.lsi_dispatcher >| lsi_dispatcher.log 2>&1 &", just_issue=True)
    tpo.setenv("PYRO_NS_HOST", PYRO_NS_HOST)
    tpo.setenv("PYRO_SERIALIZERS_ACCEPTED", "pickle")
    tpo.setenv("PYRO_SERIALIZER", "pickle")
    gh.run("{PYTHON} -m Pyro4.naming --host {PYRO_NS_HOST} --port 9090 >| run-pyro-host-port.log 2>&1 &", just_issue=True) 
    gh.run("{PYTHON} -m gensim.models.lsi_dispatcher >| lsi_dispatcher.log 2>&1 &", just_issue=True)
    gh.run("sleep 5")

    # Start remote clients
    lsi_worker_script_script = "/tmp/run-single-lsi-worker.sh"
    gh.write_lines(lsi_worker_script_script, 
                   ["export PYRO_SERIALIZERS_ACCEPTED=pickle",
                    "export PYRO_SERIALIZER=pickle",
                    "export PYRO_NS_HOST=" + PYRO_NS_HOST,
                    "python -m gensim.models.lsi_worker >| /tmp/run-single-lsi-worker.log 2>&1 &"])
    copy_file_to_workers(remote_workers, lsi_worker_script_script)
    run_command_on_workers(remote_workers, tpo.format("source {lsi_worker_script_script}"))

    # Show the pyro clients running
    tpo.debug_print("active pyro clients:\n%s\n" % gh.run("PYRO_NS_HOST={PYRO_NS_HOST}  {PYTHON}  -m Pyro4.nsc list"), 4)
    return


def stop_remote_workers(remote_workers):
    """Stop lsi_worker interface on each of the REMOTE_WORKERS along with local dispatcher"""
    run_command_on_workers(remote_workers, "csh -b /tmp/kill_em.sh -p lsi_worker")
    gh.run("csh -b /tmp/kill_em.sh -p lsi_dispatcher")
    return

#-------------------------------------------------------------------------------

def main():
    """Entry point for script"""
    tpo.debug_print("main(): sys.argv=%s" % sys.argv, 4)

    # Check command-line arguments
    env_options = tpo.formatted_environment_option_descriptions()
    usage_description = tpo.format("""
Performs Latent Semantic Analysis (LSA) over gensim corpus file, optionally 
performing Latent Dirichlet Allocation (LDA) as well.

Notes:
- The corpus should be in matrix market (mm) format, such as via gensim_test.py
  (or via gensim/scripts/make_wiki script.py over Wikipedia dump).
- Requires TF/IDF version of the corpus in BASENAME.tfidf.mm.
- Latent Semantic Indexing (LSI) is same as Latent Semantic Analysis (LSA).
- The following environment options are available:
	{env_options}
""")
    # Check command-line arguments
    # TODO: make sure each argument supported via external API
    # TODO: use epilog for notes (and add example as well)
    parser = argparse.ArgumentParser(description=usage_description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    #
    parser.add_argument("--LSA", default=True, action='store_true', help="Run LSA (default)")
    parser.add_argument("--LSI", dest='LSA', action='store_true', help="Alias for --LSA")
    parser.add_argument("--skip-LSA", dest='LSA', action='store_false', help="Skip LSA processing")
    parser.add_argument("--skip-LSI", dest='LSA', action='store_false', help="Alias for --skip-LSA")
    parser.add_argument("--LDA", default=False, action='store_true', help="Run LDA as well")
    parser.add_argument("--skip-LDA", dest='LDA', action='store_false', help="Skip LDA (default)")
    parser.add_argument("--tfidf", default=False, action='store_true', help="Use TF/IDF version of corpis")
    parser.add_argument("--similarity", default=False, action='store_true', help="Derive similarity data")
    parser.add_argument("--output-basename", default="", help="Basename to use for output (by default input file without .txt extension)")
    parser.add_argument("--save", default=False, action='store_true', help="Save models to disk")
    parser.add_argument("--preprocess", default=False, action='store_true', help="Preprocess the corpus (via gensim_test.py)")
    parser.add_argument("--reuse-output-files", default=False, action='store_true', help="Reuse intermediary output files")
    parser.add_argument("--transformed", default=False, action='store_true', help="Also save LSA transformed version of corpus")
    parser.add_argument("--load", default=False, action='store_true', help="Load models from disk")
    parser.add_argument("--print", default=True, action='store_true', help="Print LSA info on standard output (e.g., top topics)")
    parser.add_argument("--skip-print", dest='print', default=False, action='store_true', help="Don't print LSA info.")
    parser.add_argument("--stochastic", default=False, action='store_true', help="Use multi-pass stochostic (LSA only)")
    parser.add_argument("--num-topics", type=int, default=NUM_TOPICS, help="Number of topics (i.e., reduced dimensions)")
    parser.add_argument("--similar-docs-of", default="", help="Show similar documents for list of document ID's (or * for all)")
    parser.add_argument("--max-similar", type=int, default=MAX_SIMILAR, help="Maximum number of similar documents to return")
    parser.add_argument("--verbose", default=False, action='store_true', help="Verbose output mode (e.g., resolve term ID's)")
    parser.add_argument("--docid-filename", default=None, help="Filename with document ID's")
    parser.add_argument("--prune-dictionary", default=False, action='store_true', help="Prune dictionary prior to LSA")
    parser.add_argument("--num-documents", type=int, default=-1, help="Number of documents to process (e.g., when using augmentation corpus)")
    parser.add_argument("--distributed", default=False, action='store_true', help="Run distributed computations")
    parser.add_argument("--setup-workers", default=False, action='store_true', help="Install necessary packages on worked")
    parser.add_argument("--start-workers", default=False, action='store_true', help="Start LSI client interface on workers")
    parser.add_argument("--remote-workers", default="", help="List of hosts serving as workers")
    #
    # note: basename is positional argument
    # TODO: rework --basename to be like gensim_test.py's --output-basename
    parser.add_argument("basename", default="", help="Basename for input data (e.g., enwiki-20140402-pages-articles.gensim-prep via gensim.scripts.make_wiki script)")
    #
    args = vars(parser.parse_args())
    tpo.debug_print("args = %s" % args, 5)
    basename = args['basename']
    output_basename = args['output_basename'] or basename
    run_lda = args['LDA']
    run_lsa = args['LSA']
    load = args['load']
    save = args['save']
    transformed = args['transformed']
    use_tfidf = args['tfidf']
    print_info = args['print']
    use_one_pass = (not args['stochastic'])
    num_topics = args["num_topics"]
    derive_similarity = args['similarity']
    max_similar = args['max_similar']
    num_documents = args['num_documents']
    source_similar_docs = args['similar_docs_of'].replace(",", " ").split()
    verbose_output = args['verbose']
    docid_filename = args['docid_filename']
    prune_dictionary = args['prune_dictionary']
    distributed = args['distributed']
    setup_workers = args['setup_workers']
    start_workers = args['start_workers']
    remote_workers = args['remote_workers'].split()

    # Make sure document-term matrix exists
    # TODO: have option to use regular corpus (i.e., non-TFIDF) for comparison purposes
    matrix_ext = ".tfidf.mm" if use_tfidf else ".bow.mm"
    matrix_file = (basename + matrix_ext)
    if args["preprocess"]: 
        if not (args['reuse_output_files'] and (gh.non_empty_file(matrix_file))):
            tfidf_option = "--tfidf" if use_tfidf else ""
            misc_options = ""
            if print_info:
                misc_options += "--print "
            if verbose_output:
                misc_options += "--verbose "
            gh.run("{PYTHON} -m gensim_test  {tfidf_option}  --save  {misc_options}  {basename}.txt")
    if (not os.path.exists(matrix_file)):
        tpo.print_stderr("Error: Unable to find '%s': first preprocess the text (see gensim_test.py)." % matrix_file)
        sys.exit(1)
    gh.assertion(os.path.exists(basename + '.wordids.txt.bz2'))

    # Enable logging if debugging
    if (tpo.debugging_level()):
        level = logging.INFO if (tpo.debug_level < 4) else logging.DEBUG
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=level)

    # Optionally set random seed
    if RANDOM_SEED != -1:
        tpo.debug_format("Setting random seed to {RANDOM_SEED}")
        numpy.random.seed(RANDOM_SEED)

    # Load in the corpus along with id->word mapping (the dictionary), one of the results of the preprocessing.
    # HACK: (plain) corpus loaded just for dictionary when tfidf version used
    # NOTE: CORPUS_BASENAME and TDIDF_CORPUS overrides provided to facilitate run_relatedness_checks.py
    # TODO: rework so that command-line option used environment as default (not vice versa)
    corpus_basename = tpo.getenv_text("CORPUS_BASENAME", basename)
    corpus_data = CorpusData()
    corpus_data.load(corpus_basename)
    tpo.debug_print("corpus_data: type=%s value=%s" % (type(corpus_data), corpus_data), 5)
    corpus = corpus_data.mm
    dictionary = corpus_data.dictionary

    # Optionally prune low and high frequency terms from dictioanry
    if (prune_dictionary):
        if ((dictionary.num_docs == 0) and (NUM_DOCS > 0)):
            tpo.debug_print("HACK: performing dictionary fixup for number of documents", 4)
            dictionary.num_docs = NUM_DOCS
        dictionary.filter_extremes()

    # Optionally load in TFIDF-weighted version of corpus (raw frequencies converted into TF/IDF scores)
    if (use_tfidf):
        tdidf_corpus_filename = tpo.getenv_text("TDIDF_CORPUS", (output_basename + '.tfidf.mm'))
        tfidf_corpus = gensim.corpora.MmCorpus(tdidf_corpus_filename)
        tpo.debug_print("tfidf_corpus: type=%s value=%s" % (type(tfidf_corpus), tfidf_corpus), 5)
        corpus = tfidf_corpus


    # Perform LSA
    if (run_lsa):
        if (setup_workers):
            setup_remote_workers(remote_workers)
        if (start_workers):
            start_remote_workers(remote_workers)
        # extract 400 LSI topics, using the default one-pass algorithm
        if (load):
            lsa_model_filename = tpo.getenv_text("LSA_MODEL", basename + ".lsa")
            lsa = tpo.load_object(lsa_model_filename)
        else:
            lsa = gensim.models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, onepass=use_one_pass, distributed=distributed)
            tpo.trace_object(lsa, 6, "lsimodel")
            tpo.trace_object(lsa.projection, 7, "lsimodel.projection")

            # print the most contributing words (both positively and negatively) for each of the first N topics
            if (print_info):
                lsa.print_topics(TOP_TOPICS)
        tpo.debug_format("lsa={lsa}", 4)
        if (save):
            if (not (load and gh.non_empty_file(output_basename + ".lsa"))):
                tpo.store_object(output_basename + ".lsa", lsa)
            if (transformed):
                transformed_file = (output_basename + ".lsa.mm")
                gensim.corpora.MmCorpus.serialize(transformed_file, (gensim.matutils.unitvec(vec) for vec in lsa[corpus]))
        if (start_workers):
            stop_remote_workers(remote_workers)

    # Optionally set last document to consider for comparisons
    if num_documents != -1:
        tpo.debug_format("Setting number of documents for comparison to {nd}", 4,
                         nd=num_documents)
        corpus.num_docs = num_documents

    # Determine similarity model
    if (derive_similarity or source_similar_docs):
        lsa_sim_index_filename = tpo.getenv_text("LSA_SIM_INDEX", output_basename + ".lsa.sim_index")
        sim = SimilarDocumentByLSA(corpus=corpus, dictionary=dictionary, lsa=lsa, index_file=lsa_sim_index_filename, verbose_output=verbose_output, max_similar=max_similar, docid_filename=docid_filename)
        # Precompute similarities
        if not source_similar_docs:
            sim.derive_all_similarities()

    # Show similar documents
    # Note: This first creates a "similarity index" over LSA-transformed document-term space
    # and then performs lookup over each query document's vector likewise transformed via LSA.
    # TODO: extend similarity support in gensim_test.py to operate over transformed matrices (e.g., from lsa model file)
    if source_similar_docs:
        assert(run_lsa)
        if (source_similar_docs == ['*']):
            similar_doc_info = sim.find_all_similar()
        else:
            similar_doc_info = []
            for docid in source_similar_docs:
                similar_doc_info.append((docid, sim.find(docid)))
        if tpo.verbose_debugging():
            similar_doc_info = list(similar_doc_info)
            tpo.debug_format("similar_doc_info={similar_doc_info}")
        for (docid, similar_docs) in similar_doc_info:
            similar_docs = [(d, tpo.round_num(score)) for (d, score) in similar_docs]
            print("Documents similar to %s: %s" % (docid, similar_docs))
            gh.assertion(len(similar_docs) > 0)
        if (save):
            sim.save(output_basename + '.lsa.sim_index')


    # Optionally perform LDA
    if (run_lda):
        # Extract 100 LDA topics, using 1 pass and updating once every 1 chunk (10,000 documents)
        # TODO: add more parameters (e.g., chunksize and passes)
        if (load):
            lda = tpo.load_object(basename + ".lda")
        else:
            lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                                                  update_every=1, chunksize=10000, passes=1)
            # Print the most contributing words for randomly selected topics
            # Note: with LDA the topics aren't ordered (unlike LSA)
            lda.print_topics(TOP_TOPICS)
        tpo.debug_format("lda={lda}", 4)
        if (save):
            if (not (load and gh.non_empty_file(output_basename + ".lda"))):
                tpo.store_object(output_basename + ".lda", lda)
            if (source_similar_docs):
                sim.save(output_basename + '.index')

    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
