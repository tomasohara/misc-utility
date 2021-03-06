#! /usr/bin/env python
#
# Support for performing Term Frequency (TF) Inverse Document Frequency (IDF)
# using ngrams. This is provides a wrapper class around the tfidf package
# by elzilrac (https://github.com/elzilrac/tf-idf).
#
# For details on computations, see following Wikipedia pages:
#    https://en.wikipedia.org/wiki/Tf-idf
#    https://en.wikipedia.org/wiki/N-gram.
#
# Note:
# - This provides the wrapper class ngram_tfidf_analysis around tfidf for use
#   in applications like Visual Diff Search (VDS) that use text from external sources.
# - See compute_tfidf.py for computing tfidf over files.
#
# TODO:
# - Add filtering (e.g., subsumption, all numbers).
#
#

## TODO: fix description (e.g., add pointer to VDS code)
"""TF-IDF using phrasal terms via ngram analysis"""

# Standard packages
## import math
## import os
import re
import sys

# Installed packages
import tfidf
from tfidf.corpus import Corpus as tfidf_corpus
from tfidf.preprocess import Preprocessor as tfidf_preprocessor

# Local packages
import debug
import system
import tpo_common as tpo

PREPROCESSOR_LANG = system.getenv_text("PREPROCESSOR_LANG", "english")
# NOTE: add MIN_NGRAM_SIZE (e.g., 2) as alternative to ALL_NGRAMS (implies 1)
# TODO: NGRAM_SIZE => MAX_NGRAM_SIZE
NGRAM_SIZE = system.getenv_int("NGRAM_SIZE", 2)
MIN_NGRAM_SIZE = system.getenv_int("MIN_NGRAM_SIZE", 2)
ALL_NGRAMS = system.getenv_boolean("ALL_NGRAMS", False)
USE_NGRAM_SMOOTHING = system.getenv_boolean("USE_NGRAM_SMOOTHING", False)
DEFAULT_TF_WEIGHTING = 'basic'
TF_WEIGHTING = system.getenv_text("TF_WEIGHTING", DEFAULT_TF_WEIGHTING)
DEFAULT_IDF_WEIGHTING = 'smooth' if USE_NGRAM_SMOOTHING else 'basic'
IDF_WEIGHTING = system.getenv_text("IDF_WEIGHTING", DEFAULT_IDF_WEIGHTING)
MAX_TERMS = system.getenv_int("MAX_TERMS", 100)
ALLOW_NGRAM_SUBSUMPTION = system.getenv_boolean("ALLOW_NGRAM_SUBSUMPTION", False)
## TODO: ALLOW_NGRAM_OVERLAP = system.getenv_boolean("ALLOW_NGRAM_SUBSUMPTION", False)
ALLOW_NUMERIC_NGRAMS = system.getenv_boolean("ALLOW_NUMERIC_NGRAMS", False)

try:
    # Note major and minor revision values are assumed to be integral
    major_minor = re.sub(r"^(\d+\.\d+).*", r"\1", tfidf.__version__)
    TFIDF_VERSION = float(major_minor)
except:
    TFIDF_VERSION = 1.0
    system.print_stderr("Exception in main: " + str(sys.exc_info()))
assert(TFIDF_VERSION > 1.0)

            
class ngram_tfidf_analysis(object):
    """Class for performs TF-IDF over ngrams and returning sorted list"""

    def __init__(self, pp_lang=PREPROCESSOR_LANG, ngram_size=NGRAM_SIZE, *args, **kwargs):
        """Class constructor (mostly for tracing)"""
        # TODO: add option for stemmer; add all_ngrams and min_ngram_size to constructor
        debug.trace_fmtd(5, "args={a} kwargs={k}", a=args, k=kwargs)
        if pp_lang is None:
            pp_lang = PREPROCESSOR_LANG
        self.pp = tfidf_preprocessor(language=pp_lang,
                                     gramsize=ngram_size,
                                     min_ngram_size=MIN_NGRAM_SIZE,
                                     all_ngrams=ALL_NGRAMS,
                                     stemmer=lambda x: x)
        self.corpus = tfidf_corpus(gramsize=ngram_size,
                                   min_ngram_size=MIN_NGRAM_SIZE,
                                   all_ngrams=ALL_NGRAMS,
                                   language=pp_lang,
                                   preprocessor=self.pp)
        super(ngram_tfidf_analysis, self).__init__(*args, **kwargs)

    def add_doc(self, text, doc_id=None):
        """Add document TEXT to collection with key DOC_ID, which defaults to order processed (1-based)"""
        if doc_id is None:
            doc_id = str(len(self.corpus) + 1)
        self.corpus[doc_id] = text

    def get_doc(self, doc_id):
        """Return document data for DOC_ID"""
        return self.corpus[doc_id]

    def get_top_terms(self, doc_id, tf_weight=TF_WEIGHTING, idf_weight=IDF_WEIGHTING, limit=MAX_TERMS):
        """Return list of (term, weight) tuples for DOC_ID up to LIMIT count, using TF_WEIGHT and IDF_WEIGHT schemes
        Notes:
        - TF_WEIGHT in {basic, binary, freq, log, norm_50}
        - IDF_WEIGHT in {basic freq, max, prob, smooth}
        - The top ngrams omit blanks and other relics of tokenization
        - Lower weighted ngrams are omitted if subsumed by higher (or vice versa)
        """
        # Get objects for top terms
        # ex: top_terms=[CorpusKeyword(term=<tfidf.dockeyword.DocKeyword object at 0x7f08b43bf550>, ngram=u'patuxent river', score=0.0015548719642054984), ... CorpusKeyword(term=<tfidf.dockeyword.DocKeyword object at 0x7f08b43cf110>, ngram=u'afognak native corporation', score=0.0009894639772216809)]
        # Get twice as many top terms to display to account for filtering
        # TODO: keep track of how often too few terms shown
        top_terms = self.corpus.get_keywords(document_id=doc_id,
                                             tf_weight=tf_weight,
                                             idf_weight=idf_weight,
                                             limit=(2 * limit))
        debug.trace_fmtd(7, "top_terms={tt}", tt=top_terms)
        # Skip empty tokens due to spacing and to punctuation removal (e.g, " ")
        temp_top_term_info = [(k.ngram, k.score) for k in top_terms if k.ngram.strip()]
        # Put spaces around ngrams to aid in subsumption tests
        spaced_ngrams = [(" " + ngram + " ") for (ngram, _score) in temp_top_term_info]
        debug.trace_fmtd(7, "spaced_ngrams={sn}", sn=spaced_ngrams)
        top_term_info = []
        for (i, (ngram, score)) in enumerate(temp_top_term_info):
            
            if (not ngram.strip()):
                debug.trace_fmt(5, "Omitting invalid ngram '{ng}'", ng=ngram)
                continue
            if ((not ALLOW_NUMERIC_NGRAMS) and all([tpo.is_numeric(token) for token in ngram.split()])):
                debug.trace_fmt(5, "Omitting numeric ngram '{ng}'", ng=ngram)
                continue
            
            # Subsumption (e.g., "new york" in "new york city")
            # TODO: Add similar overlap check (e.g., "new york city" and "york city subway"); record ngram offsets to facilitate contiguity tests
            include = True
            for (j, other_spaced_ngram) in enumerate(spaced_ngrams):
                if ((i > j) and ((spaced_ngrams[i] in other_spaced_ngram)
                                 or (other_spaced_ngram in spaced_ngrams[i]))):
                    debug.trace_fmt(5, "Omitting lower-weigted ngram '{ng2}' in subsumption with '{ng1}'", ng1=other_spaced_ngram, ng2=spaced_ngrams[i])
                    include = False
                    break
            if not include:
                continue

            # OK
            top_term_info.append((ngram, score))
            if (len(top_term_info) == limit):
                break
        # Sanity check on number of terms displayed
        num_terms = len(top_term_info)
        if (num_terms < limit):
            debug.trace_fmt(3, "Warning: only {n} terms shown (of {m} max)",
                            n=num_terms, m=limit)
        debug.trace_fmtd(6, "top_term_info={tti}", tti=top_term_info)
        return top_term_info

    def get_ngrams(self, text):
        """Returns generator with ngrams in TEXT"""
        ## BAD return self.pp.yield_keywords(text)
        ngrams = []
        gen = self.pp.yield_keywords(text)
        more = True
        while (more):
            ## DEBUG: debug.trace_fmtd(6, ".")
            try:
                ngrams.append(gen.next().text)
            except StopIteration:
                more = False
        return ngrams
    
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    # TODO: add simple test (e.g., over tfidf source)
    system.print_stderr("Error: not intended for command-line use")
