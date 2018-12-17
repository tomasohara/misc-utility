#! /usr/bin/env python
#
# compute_tfidf.py: compute Term Frequency Inverse Document Frequency (TF-IDF)
# for a collection of documents. See https://en.wikipedia.org/wiki/Tf-idf.
#
# TODO:
# - Show examples.
# - Have option for producing document/term matrix.
# - Have option to show TF and IDF values.
# - Add option for IDF weighting: max, prob, and basic.
# - Add option for TF weighting: log, norm_50 (double normalized), binary, and basic.
# - Add option to omit TF,IDF and IDF fields (e.g., in case just interested in frequency counts).
# - Add option to put term at end, so numeric fields are aligned.
# - -and/or- have ** max-output-term-length option and right pad w/ spaces.
#
# Note:
# - This script is just for running tfidf over text files.
# - See ngram_tfdif.py for a wrapper class around tfidf for use in applications
#   like Visual Diff Search that generate the text dynamically.
#
# Copyright (c) 2018 Scrappycito, LLC
#


"""Extract text from common document types"""

# Standard packages
## TODO: import re
import re
import sys

# Installed packages
# TODO: require version 1.1 with TPO hacks
import tfidf
from tfidf.corpus import Corpus as tfidf_corpus
from tfidf.preprocess import Preprocessor as tfidf_preprocessor

# Local packages
import debug
import system


# Determine environment-based options
DEFAULT_NUM_TOP_TERMS = system.getenv_int("NUM_TOP_TERMS", 10)
DEFAULT_NGRAM_SIZE = system.getenv_int("NGRAM_SIZE", 1)
MIN_NGRAM_SIZE = system.getenv_int("MIN_NGRAM_SIZE", DEFAULT_NGRAM_SIZE)
IDF_WEIGHTING = system.getenv_text("IDF_WEIGHTING", 'basic')
TF_WEIGHTING = system.getenv_text("TF_WEIGHTING", 'basic')

# Option names and defaults
NGRAM_SIZE = "--ngram-size"
NUM_TOP_TERMS = "--num-top-terms"
## OLD: NGRAM_SMOOTHING = "--use-ngram-smoothing"
SHOW_SUBSCORES = "--show-subscores"
SHOW_FREQUENCY = "--show-frequency"

#...............................................................................

def show_usage_and_quit():
    """Show command-line usage for script and then exit"""
    # TODO: make 
    usage = """
Usage: {prog} [options] file1 file2 ...

Options: [--help] [{ngrams}=N] [{top_terms}=N] [{subscores}] [{frequencies}]

Notes:
- Derives TF-IDF for set of documents, using single word tokens (unigrams),
  by default. 
- Use following environment options:
      DEFAULT_NUM_TOP_TERMS ({default_topn})
      DEFAULT_NGRAM_SIZE ({default_ngram})
      MIN_NGRAM_SIZE ({min_ngram_size})
      TF_WEIGHTING ({tf_weighting}): {{log, norm_50, binary, basic, freq}}
      IDF_WEIGHTING ({idf_weighting}): {{smooth, max, prob, basic, freq}}
""".format(prog=sys.argv[0], ngrams=NGRAM_SIZE, top_terms=NUM_TOP_TERMS, subscores=SHOW_SUBSCORES, frequencies=SHOW_FREQUENCY, default_topn=DEFAULT_NUM_TOP_TERMS, default_ngram=DEFAULT_NGRAM_SIZE, min_ngram_size=MIN_NGRAM_SIZE, tf_weighting=TF_WEIGHTING, idf_weighting=IDF_WEIGHTING)
    print(usage)
    sys.exit()

def main():
    """Entry point for script"""
    args = sys.argv[1:]
    debug.trace_fmtd(5, "main(): args={a}", a=args)

    # Parse command-line arguments
    i = 0
    ngram_size = DEFAULT_NGRAM_SIZE
    num_top_terms = DEFAULT_NUM_TOP_TERMS
    ## OLD: use_ngram_smoothing = False
    show_subscores = False
    show_frequency = False
    while ((i < len(args)) and args[i].startswith("-")):
        option = args[i]
        debug.trace_fmtd(5, "arg[{i}]: {opt}", i=i, opt=option)
        if (option == "--help"):
            show_usage_and_quit()
        elif (option == NGRAM_SIZE):
            i += 1
            ngram_size = int(args[i])
        elif (option == NUM_TOP_TERMS):
            i += 1
            num_top_terms = int(args[i])
        ## OLD: subsumed by IDF_WEIGHTING
        ## elif (option == NGRAM_SMOOTHING):
        ##     use_ngram_smoothing = True
        elif (option == SHOW_SUBSCORES):
            show_subscores = True
        elif (option == SHOW_FREQUENCY):
            show_frequency = True
            # Make sure TF-IDF package supports occurrence counts for TF
            tfidf_version = 1.0
            try:
                # Note major and minor revision values assumed to be integral
                major_minor = re.sub(r"^(\d+\.\d+).*", r"\1", tfidf.__version__)
                tfidf_version = float(major_minor)
            except:
                system.print_stderr("Exception in main: " + str(sys.exc_info()))
            assert(tfidf_version >= 1.2)
        else:
            sys.stderr.write("Error: unknown option '{o}'\n".format(o=option))
            show_usage_and_quit()
        i += 1
    args = args[i:]
    if (len(args) < 1):
        show_usage_and_quit()
    if ((len(args) < 2) and (not show_frequency)):
        ## TODO: only issue warning if include-frequencies not specified
        system.print_stderr("Warning: TF-IDF not relevant with only one document")

    # Initialize Tf-IDF module
    my_pp = tfidf_preprocessor(language='english', gramsize=ngram_size, min_ngram_size=MIN_NGRAM_SIZE, all_ngrams=False, stemmer=lambda x: x)
    corpus = tfidf_corpus(gramsize=ngram_size, min_ngram_size=MIN_NGRAM_SIZE, all_ngrams=False, preprocessor=my_pp)

    # Process each of the arguments
    doc_filenames = {}
    for i, filename in enumerate(args):
        doc_id = str(i + 1)
        doc_text = system.read_entire_file(filename)
        corpus[doc_id] = doc_text
        doc_filenames[doc_id] = filename
    debug.trace_object(7, corpus, "corpus")

    # Derive headers
    headers = ["term"]
    if show_frequency:
        headers += ["TFreq", "DFreq"]
    if show_subscores:
        headers += ["TF", "IDF"]
    headers += ["TF-IDF"]

    # Output the top terms per document with scores
    # TODO: change the IDF weighting
    ## OLD: IDF_WEIGHTING = 'smooth' if use_ngram_smoothing else 'basic'
    ## BAD: for doc_id in corpus:
    for doc_id in corpus.keys():
        print("{id}. {filename}".format(id=doc_id, filename=doc_filenames[doc_id]))
        print("\t".join(headers))

        # Get ngrams for document and calculate overall score (TF-IDF).
        # Then print each in tabular format (e.g., "et al   0.000249")
        top_term_info = corpus.get_keywords(document_id=doc_id,
                                            idf_weight=IDF_WEIGHTING,
                                            limit=num_top_terms)
        for (term, score) in [(t.ngram, t.score)
                              for t in top_term_info if t.ngram.strip()]:
            # Get scores including component values (e.g., IDF)
            scores = []
            if show_frequency:
                scores.append(corpus[doc_id].tf_freq(term))
                scores.append(corpus.df_freq(term))
            if show_subscores:
                scores.append(corpus[doc_id].tf(term, tf_weight=TF_WEIGHTING))
                scores.append(corpus.idf(term))
            scores.append(score)

            # Print term and rounded scores
            rounded_scores = [str(system.round_num(s)) for s in scores]
            print("{t}\t{rs}".format(t=system.to_utf8(term),
                                     rs="\t".join(rounded_scores)))
        print("")

    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
