#! /usr/bin/env python
#
# Program to filter word lists based on ngram models. The initial purpose
# is to support filtering of proper names from spelling correction rewrite
# rules, both for unconditional replacements and contextual rewrites. For
# efficiency, lookup for ngrams larger than 2 are based on pairs of attested
# bigrams starting with an attested unigram.
#
# The required supporting data consists of the ngram tables (e.g., derived from 
# large lists of proper names). Given an input string of words, the program
# will output subphrases that correspond to entries in the word lists.
#
# Notes:
# - See juju_prototype/keyword_analysis.py for ngram support in the context
#   of job description analysis (also suitable for general text).
# - See train_language_model.py and kenlm_example.py for more general approach
#   based on language modeling and smoothing.
# - Uses token and bigram frequency files prepared elsewhere.
#
# TODO:
# - Have option to remove ngram subprases can be removed from the input.
#
#-------------------------------------------------------------------------------
# Sample input and output
#
#   $ head _dog_fleas.*
#   ==> _dog_fleas.bi <==
#   my dog	10
#   dog has	5
#   has fleas	1
#   
#   ==> _dog_fleas.txt <==
#   My dog has fleas very fat. Poor dog of mine!
#   
#   ==> _dog_fleas.uni <==
#   dog	        1000
#   fleas	25
#
#   $ python -m ngram_filter _dog_fleas.uni _dog_fleas.bi < _dog_fleas.txt 
#   My dog has fleas very fat. Poor dog of mine! => ['my dog has fleas', 'dog']
#

"""Extracts ngrams in input based on unigram and bigram frequencies"""

import argparse
## import fileinput
import sys
import re
import tpo_common as tpo

## TODO: MIN_UNIGRAM_FREQ = tpo.getenv_integer("MIN_UNIGRAM_FREQ", 1)
MIN_BIGRAM_FREQ = tpo.getenv_integer("MIN_BIGRAM_FREQ", 1)

def create_frequency_hash(filename):
    """Create lookup hash from term mappings in filename (one pair per line); 
    uses lowercase keys (and assumed frequency of 1 if omitted)."""
    tpo.debug_print("create_frequency_hash(%s)" % filename, 4)
    lookup_hash = {}
    with open(filename) as f:
        line_num = 0
        for line in f:
            line_num += 1
            fields = line.lower().split("\t")
            term = fields[0]
            freq = fields[1] if (len(fields) > 1) else 1
            if len(fields) > 2:
                tpo.debug_format("Warning: Ignoring {d} at {f}:{n}: {l}", 3,
                                 d=fields[2:], n=line_num, f=filename, l=line)
            lookup_hash[term] = int(freq)
    tpo.debug_format("create_frequency_hash => {h}", 7, h=lookup_hash)
    return lookup_hash

def main():
    """Entry point for script"""
    # Parse arguments: unigrams bigrams
    # TODO: accept optional filename for input
    parser = argparse.ArgumentParser()
    parser.add_argument("unigram_file", help="file with word token frequencies")
    parser.add_argument("bigram_file", help="file with word pair frequencies")
    args = parser.parse_args()
    tpo.debug_print("args: %s" % args, 4)
    unigram_hash = create_frequency_hash(args.unigram_file)
    bigram_hash = create_frequency_hash(args.bigram_file)

    # Output portions of input matched by ngram models
    line_num = 0
    for line in sys.stdin.readlines():
        line = line.strip()
        line_num += 1
        ## TODO: tpo.debug_print("L%d: %s" % (fileinput.filelineno(), line), 5)
        tpo.debug_print("L%d: %s" % (line_num, line), 5)
        ## OLD: terms = re.split("\W+", line.lower())
        terms = [t for t in re.split(r"\W+", line.lower()) if t]
        tpo.debug_print("terms: %s" % terms, 5)
        subphrases = []
       
        # Check whether adjancent terms are recognized bigrams
        starts_bigram = []
        for i in range(0, len(terms) - 1):
            pair = terms[i] + " " + terms[i + 1]
            is_start = False
            if pair in bigram_hash:
                tpo.debug_format("bigram_hash[{p}]: {f}", 5,
                                 p=pair, f=bigram_hash[pair])
                is_start = (bigram_hash[pair] >= MIN_BIGRAM_FREQ)
            starts_bigram.append(is_start)
        tpo.debug_format("starts bigram: {s}", 6,
                         s=zip(terms, starts_bigram + [-1]))
        tpo.debug_format("bigrams: {b}", 5,
                         b=[(terms[s] + "-" + terms[s + 1]) for s in 
                            range(0, len(starts_bigram)) if starts_bigram[s]])
        tpo.debug_format("unigrams: {u}", 5, 
                         u=[t for t in terms if t in unigram_hash])

        # Find largest match subphrases: multiword spans with all consecutive
        # bigrams attested or single word span for attested unigram
        # TODO: factor in bigram/unigram frequencies to approximate phrasal ones
        start = 0
        while start < len(terms):
            sublen = 1
            while (((start + sublen - 1) < len(starts_bigram)) 
                   and (starts_bigram[start + sublen - 1])):
                sublen += 1
            tpo.debug_format("start: {st}; sublen: {len}; sub={sub}", 5,
                             st=start, len=sublen, 
                             sub=" ".join(terms[start:(start + sublen)]))
            if sublen > 1:
                subphrases.append(" ".join(terms[start:(start + sublen)]))
                start += sublen
            else:
                include = False
                if terms[start] in unigram_hash:
                    ## TODO: if (unigram_hash[terms[start]] > MIN_UNIGRAM_FREQ)
                    include = True
                    if include:
                        subphrases.append(terms[start])
                start += 1

        # Output matching subphrases
        print("%s => %s" % (line, subphrases))

    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
