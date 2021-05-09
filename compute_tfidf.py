#! /usr/bin/env python
#
# compute_tfidf.py: compute Term Frequency Inverse Document Frequency (TF-IDF)
# for a collection of documents. See https://en.wikipedia.org/wiki/Tf-idf.
#
# TODO:
# - *** Clarify confusing options regarding ngram size ***
# - ** Add sanity checks (e.g., soft assertions). **
# - Show examples.
# - Have option for producing document/term matrix.
# - Have option to show TF and IDF values.
# - Add option for IDF weighting: max, prob, and basic.
# - Add option for TF weighting: log, norm_50 (double normalized), binary, and basic.
# - Add option to omit TF,IDF and IDF fields (e.g., in case just interested in frequency counts).
# - Add option to put term at end, so numeric fields are aligned.
# - -and/or- Have ** max-output-term-length option and right pad w/ spaces.
# - See if IDF calculation should for 0 if just one document occurrence.
#
# Note:
# - This script is just for running tfidf over text files.
# - See ngram_tfdif.py for a wrapper class around tfidf for use in applications
#   like Visual Diff Search that generate the text dynamically.
#

## TODO: update/fix/refine description (e.g., add mention of file1, ..., fileN or of file.csv)
"""Extract text from common document types"""

# Standard packages
## TODO: import re
import csv
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
# TODO: * add env-var comments to better distinguish confusing ngram options!
DEFAULT_NGRAM_SIZE = system.getenv_int("DEFAULT_NGRAM_SIZE", 1)
NGRAM_SIZE_PROPER = system.getenv_int("NGRAM_SIZE", DEFAULT_NGRAM_SIZE)
debug.assertion(NGRAM_SIZE_PROPER >= DEFAULT_NGRAM_SIZE)
MIN_NGRAM_SIZE = system.getenv_int("MIN_NGRAM_SIZE", DEFAULT_NGRAM_SIZE)
IDF_WEIGHTING = system.getenv_text("IDF_WEIGHTING", "basic")
TF_WEIGHTING = system.getenv_text("TF_WEIGHTING", "basic")
DELIMITER = system.getenv_text("DELIMITER", ",")

# Option names and defaults
## OLD: NGRAM_SIZE = "--ngram-size"
NGRAM_SIZE_OPTION = "--ngram-size"
NUM_TOP_TERMS = "--num-top-terms"
## OLD: NGRAM_SMOOTHING = "--use-ngram-smoothing"
SHOW_SUBSCORES = "--show-subscores"
SHOW_FREQUENCY = "--show-frequency"
CSV = "--csv"

#...............................................................................

def show_usage_and_quit():
    """Show command-line usage for script and then exit"""
    # TODO: make ???
    # TODO: see why --csv showing up as '{--csv}' (i.e., extraneous brace and quotes)
    usage = """
Usage: {prog} [options] file1 ... fileN

Options: [--help] [{ngrams_args}=N] [{top_terms}=N] [{subscores}] [{frequencies}] [{csv}]

Notes:
- Derives TF-IDF for set of documents, using single word tokens (unigrams),
  by default. 
- By default, the document ID is the position of the file on the command line (e.g., N for fileN above). The document text is the entire file.
- However, with {csv}, the document ID is taken from the first column, and the document text from the second columns (i.e., each row is a distinct document).
- Use following environment options:
      DEFAULT_NUM_TOP_TERMS ({default_topn})
      NGRAM_SIZE ({default_ngram})
      MIN_NGRAM_SIZE ({min_ngram_size})
      TF_WEIGHTING ({tf_weighting}): {{log, norm_50, binary, basic, freq}}
      IDF_WEIGHTING ({idf_weighting}): {{smooth, max, prob, basic, freq}}
    """.format(prog=sys.argv[0], ngrams_args=NGRAM_SIZE_OPTION, top_terms=NUM_TOP_TERMS, subscores=SHOW_SUBSCORES, frequencies=SHOW_FREQUENCY, default_topn=DEFAULT_NUM_TOP_TERMS, default_ngram=DEFAULT_NGRAM_SIZE, min_ngram_size=MIN_NGRAM_SIZE, tf_weighting=TF_WEIGHTING, idf_weighting=IDF_WEIGHTING, csv={CSV})
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
    csv_file = False
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
        elif (option == CSV):
            csv_file = True 
        else:
            sys.stderr.write("Error: unknown option '{o}'\n".format(o=option))
            show_usage_and_quit()
        i += 1
    args = args[i:]
    if (len(args) < 1):
        system.print_stderr("Error: missing filename(s)\n")
        show_usage_and_quit()
    if ((len(args) < 2) and (not csv_file) and (not show_frequency)):
        ## TODO: only issue warning if include-frequencies not specified
        system.print_stderr("Warning: TF-IDF not relevant with only one document")

    # Initialize Tf-IDF module
    # TODO: add warning that no-op stemmer is being used (i.e., lambda x: x)
    my_pp = tfidf_preprocessor(language='english', gramsize=ngram_size, min_ngram_size=MIN_NGRAM_SIZE, all_ngrams=False, stemmer=lambda x: x)
    corpus = tfidf_corpus(gramsize=ngram_size, min_ngram_size=MIN_NGRAM_SIZE, all_ngrams=False, preprocessor=my_pp)

    # Process each of the arguments
    doc_filenames = {}
    for i, filename in enumerate(args):
        # If CVS file, treat each row as separate document, using ID from first column and data from second
        if csv_file:
            ## TODO: with open(filename, encoding="utf-8") as fh:
            ## TODO: trap for invalid imput (e.g., wrong delimiter)
            with open(filename) as fh:
                csv_reader = csv.reader(iter(fh.readlines()), delimiter=DELIMITER, quotechar='"')
                # TODO: skip over the header line
                for (line_num, row) in enumerate(csv_reader):
                    doc_id = row[0]
                    doc_text = system.from_utf8(row[1])
                    corpus[doc_id] = doc_text
                    doc_filenames[doc_id] = filename + ":" + str(line_num)
        # Otherwise, treat entire file as document and use command-line position as the document ID
        else:
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
        print("{id} [{filename}]".format(id=doc_id, filename=doc_filenames[doc_id]))
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
