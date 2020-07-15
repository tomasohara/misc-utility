#! /usr/bin/env python
#
# This script uses Lucene to determine spell checking suggestions based either via
# an external dictionary or via an index field. It takes a list of queries as input
# (optionally with quotes), and it outputs spelling suggestions for each distinct word,
# along with an indication if the word should be replaced.
#
# Based on example in Java source:
#    lucene/suggest/src/java/org/apache/lucene/search/spell/SpellChecker.java
# Also based on PyLucene index search example:
#    http://svn.apache.org/viewvc/lucene/pylucene/trunk/samples/SearchFiles.py
# See index_table_file.py and search_table_file_index.py for related scripts.
#
# Notes:
# - Input processing ends when a blank line is encountered, so quotes should be used for
#   empty queries (e.g., if taken directly from query log).
# - An existing Lucene index is needed in order to get document frequency counts for terms.
#   Use index_table_file.py to create a dummy index from a text file.
#-------------------------------------------------------------------------------
# Sample interaction:
#
# 1. Create index of word list
#    $ INDEX_DIR=/tmp/random-query-index python -m index_table_file  tests/random10000-qiqci-query.list
# 2. Create spelling index from Lucene index
#    $ SPELL_INDEX_DIR=/tmp/random-query-spell-index INDEX_DIR=/tmp/random-query-index python -m lucene_spell_check --skip-query
# 3. Issue spelling correction checks
#    echo $'maneger\ntehnology' | SPELL_INDEX_DIR=/tmp/random-query-spell-index python -m lucene_spell_check --query
#
#-------------------------------------------------------------------------------
# TODO:
# - Add support for older versions of Lucene (e.g., lucene/suggest => contrib/spellchecker).
# - Document dependence on 'contents' field in index created by index_table_file.py.
#

"""Make spelling suggestions based using Lucene (via edit-distance and chraracter ngram filtering)"""

import os
import re
import sys
import lucene

from java.io import File
from org.apache.lucene.index import DirectoryReader, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
#
from org.apache.lucene.search.spell import SpellChecker
from org.apache.lucene.search.spell import PlainTextDictionary, LuceneDictionary, HighFrequencyDictionary
from org.apache.lucene.search.spell import JaroWinklerDistance, LevensteinDistance, LuceneLevenshteinDistance, NGramDistance

import tpo_common as tpo
import glue_helpers as gh
from tpo_common import debug_print, getenv_text, getenv_number, getenv_boolean, getenv_int, memodict
from search_table_file_index import IndexLookup
import system

#------------------------------------------------------------------------

# TODO: add descriptions (for use w/ formatted_environment_option_descriptions below)
INDEX_DIR = getenv_text("INDEX_DIR", "index")
SPELL_INDEX_DIR = getenv_text("SPELL_INDEX_DIR", "spell-index")
DICT = getenv_text("DICT", "dict.txt")
FIELD = getenv_text("FIELD", "contents")
MIN_REL_FREQ = getenv_number("MIN_REL_FREQ", 0.000001)
MIN_GAIN_RATIO = getenv_number("MIN_GAIN_RATIO", 3)
# TODO: use default of 4 once experimentation subsides
MIN_WORD_LEN = getenv_number("MIN_WORD_LEN", 1)
VERBOSE = getenv_boolean("VERBOSE", False)
VERBOSITY = getenv_int("VERBOSITY", 1 if VERBOSE else 0)
ACCURACY = getenv_number("ACCURACY", 0.75)
DYNAMIC_ACCURACY_DELTA = getenv_number("DYNAMIC_ACCURACY_DELTA", 0)
MAX_SUGGESTIONS = int(getenv_number("MAX_SUGGESTIONS", 5))
DISTANCE_MEASURE = getenv_text("DISTANCE_MEASURE", "")
STOPLIST_FILE = getenv_text("STOPLIST_FILE", "")
CHECK_STOPLIST = (STOPLIST_FILE != "")
MORPHOLOGY_FILE = getenv_text("MORPHOLOGY_FILE", "")
CHECK_MORPHOLOGY = (MORPHOLOGY_FILE != "")
WORDLIST_FILE = getenv_text("WORDLIST_FILE", "")
CHECK_WORDLIST = (WORDLIST_FILE != "")
CHECK_WN_MORPHOLOGY = getenv_boolean("CHECK_WN_MORPHOLOGY", False)
CHECK_WN_ROOTS = getenv_boolean("CHECK_WN_ROOTS", False)
MORPH_REWRITE_FILE = getenv_text("MORPH_REWRITE_FILE", "")
APPLY_MORPH_REWRITES = (MORPH_REWRITE_FILE != "")
APPLY_MORPHOLOGY = (CHECK_MORPHOLOGY or CHECK_WN_MORPHOLOGY or APPLY_MORPH_REWRITES)
MAX_MISPELLING_FREQ = getenv_number("MAX_MISPELLING_FREQ", (system.MAX_INT if APPLY_MORPHOLOGY else 10))
SKIP_LEADING_NUMS = getenv_boolean("SKIP_LEADING_NUMS", False)
ADD_WORD_FREQUENCY = getenv_text("ADD_WORD_FREQUENCY", False)
PHRASAL = getenv_boolean("PHRASAL", False, "Use phrasal search")

if CHECK_WN_MORPHOLOGY or CHECK_WN_ROOTS:
    import tpo_wordnet

#------------------------------------------------------------------------

def create_indicator_hash(filename):
    """Creates binary indicator hash based on terms in filename (one per line)"""
    debug_print("create_indicator_hash(%s)" % filename, 4)
    indicator_hash = {}
    with open(filename) as f:
        for line in f:
            term = line.lower().strip()
            indicator_hash[term] = True
    return indicator_hash


def create_lookup_hash(filename):
    """Create lookup hash from term mappings in filename (one pair per line)"""
    debug_print("create_lookup_hash(%s)" % filename, 4)
    lookup_hash = {}
    with open(filename) as f:
        line_num = 0
        for line in f:
            # TODO: make lowercase?
            line_num += 1
            fields = line.split("\t")
            if len(fields) > 1:
                lookup_hash[fields[0]] = fields[1]
            else:
                debug_print("Ignoring entry at line %d (%s): %s" % (line_num, filename, line), 3)
    return lookup_hash


def is_variant(lookup_hash, word1, word2):
    """Indicates whether the two words are related based on morhology hash"""
    variant = False
    if word1 in lookup_hash:
        variant = (lookup_hash[word1] == word2)
    elif word2 in lookup_hash:
        variant = (lookup_hash[word2] == word1)
    return variant


@memodict
def wn_get_root_word(word):
    """Wrapper around wordnet.get_root_word for sake of memoization"""
    return tpo_wordnet.get_root_word(word)


def wn_is_variant(word1, word2):
    """Determine whether the two words are related according to WordNet morphology"""
    # TODO: look into multi-argument function caching
    root1 = wn_get_root_word(word1)
    return (root1 and (root1 == wn_get_root_word(word2)))


def wn_has_wordform(wordform):
    """Whether WORDFORM known to wordnet"""
    return (wn_get_root_word(wordform) is not None)


def create_rewrite_hash(filename):
    """Create lookup hash from term mappings in filename (one pair per line)"""
    # Sample input: clerks => clerks, clerk
    debug_print("create_rewrite_hash(%s)" % filename, 4)
    rewrite_hash = {}
    with open(filename) as f:
        line_num = 0
        for line in f:
            line_num += 1
            fields = line.lower().split("=>")
            if len(fields) > 1:
                source_term = fields[0].strip()
                rewrite_terms = [t.strip() for t in fields[1].split(",")]
                # TODO: assert(source == rewrites[0])
                rewrite_hash[source_term] = rewrite_terms
            else:
                debug_print("Ignoring entry at line %d (%s): %s" % (line_num, filename, line), 3)
    return rewrite_hash


def has_morph_rewrite(rewrite_hash, word1, word2):
    """Determine whether WORD1 has WORD2 as a rewrite"""
    is_rewrite = False
    if word1.lower() in rewrite_hash:
        is_rewrite = (word2.lower() in rewrite_hash[word1.lower()])
    debug_print("has_morph_rewrite(_, %s, %s) => %s" % (word1, word2, is_rewrite), 5)
    return (is_rewrite)

#------------------------------------------------------------------------

class SpellCheck(object):
    """Class for performing queries against a Lucene index"""

    def __init__(self, spell_index_dirname=SPELL_INDEX_DIR):
        debug_print("SpellCheck.__init__(%s)" % spell_index_dirname, 4)
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        debug_print('lucene version: %s' % lucene.VERSION, 3)
        self.spell_index_dirname = spell_index_dirname
        self.speller = None
        self.is_stopword = None
        if CHECK_STOPLIST:
            self.is_stopword = create_indicator_hash(STOPLIST_FILE)
        self.is_word = None
        if CHECK_WORDLIST:
            self.is_word = create_indicator_hash(WORDLIST_FILE)
        self.derived_form = None
        if CHECK_MORPHOLOGY:
            self.derived_form = create_lookup_hash(MORPHOLOGY_FILE)
        if APPLY_MORPH_REWRITES:
            self.rewrite_table = create_rewrite_hash(MORPH_REWRITE_FILE)
        return

    def init_speller(self):
        """Initialize the Lucence SpellChecker using spell checking index SPELL_INDEX_DIRNAME"""
        debug_print("SpellCheck.init_speller(): spell_index_dirname=%s" % self.spell_index_dirname, 4)
        lucene_spell_index_dir = SimpleFSDirectory(File(self.spell_index_dirname))
        self.speller = SpellChecker(lucene_spell_index_dir)
        debug_print("speller = %s" % self.speller, 5)
        # Print off some diagnostics
        debug_print("dir(speller): %s" % dir(self.speller), 6)
        # Override distance measure
        debug_print("default StringDistance: %s" % self.speller.getStringDistance(), 5)
        if DISTANCE_MEASURE != "":
            gh.assertion(eval(DISTANCE_MEASURE) in [NGramDistance, LevensteinDistance, JaroWinklerDistance, LuceneLevenshteinDistance])
            sd = eval("%s()" % DISTANCE_MEASURE)
            self.speller.setStringDistance(sd)
        # Override accuracy threshold
        default_accuracy = self.speller.getAccuracy()
        debug_print("default accuracy: %s" % default_accuracy, 5)
        if ACCURACY != default_accuracy:
            debug_print("changing accuracy: %f" % ACCURACY, 3)
            self.speller.setAccuracy(ACCURACY)
        debug_print("indexDictionary: %s" % self.speller.indexDictionary, 5)
        return

    def create_index_dir(self):
        """Create index directory for spelling suggestion lookup"""
        debug_print("SpellCheck.create_index_dir()", 4)
        if not os.path.exists(self.spell_index_dirname):
            os.mkdir(self.spell_index_dirname)
        self.init_speller()
        return

    def perform_indexing(self, lucene_dictionary):
        """Perform the actual spelling suggestion indexing given handle to Lucene dictionary"""
        debug_print("SpellCheck.perform_indexing(%s)" % lucene_dictionary, 4)
        config = IndexWriterConfig(Version.LUCENE_CURRENT, None)
        merge = False
        self.speller.indexDictionary(lucene_dictionary, config, merge)
        return

    def index_dict(self, dictionary=DICT):
        """Create spell checking suggestion index using either plain text DICT"""
        debug_print("SpellCheck.index_dict(%s)" % dictionary, 4)
        # Create index for spell checking suggestions 
        self.create_index_dir()
        _file = File(dictionary)
        _dict = PlainTextDictionary(_file)
        debug_print("_file = %s\n_dict=%s" % (_file, _dict), 5)
        self.perform_indexing(_dict)
        return

    def index_dir(self, index_dir=INDEX_DIR, field=FIELD):
        """Create spell checking suggestion index using Lucene index in INDEX_DIR over FIELD"""
        debug_print("SpellCheck.index_dir(%s, %s)" % (index_dir, field), 4)
        # Create index for spell checking suggestions 
        self.create_index_dir()
        _dir = SimpleFSDirectory(File(index_dir))
        _reader = DirectoryReader.open(_dir)
        debug_print("num docs: %d" % _reader.numDocs(), 4)
        if MIN_REL_FREQ > 0:
            _dict = HighFrequencyDictionary(_reader, field, MIN_REL_FREQ)
        else:
            _dict = LuceneDictionary(_reader, field)
        debug_print("_dir=%s\n_reader=%s\n_dict=%s" % (_dir, _reader, _dict), 5)
        self.perform_indexing(_dict)
        return

    def run(self, index_dir=None):
        """Interactively run show spelling suggestions for queries specified on the console, optionally using document frequency to detect likely misspellings"""
        # TODO: summarize the high-level processing
        debug_print("SpellCheck.run(%s)" % index_dir, 4)

        # Optionally create function with cached results for term frequency in specified index
        if index_dir:
            # TODO: add sanity check for missing index
            index_lookup = IndexLookup(index_dir)

            @memodict
            def doc_freq(term):
                """Document frequency for TERM"""
                return index_lookup.doc_freq(term)

        # Perform other initializations
        self.init_speller()
        processed = {}

        # Process input line by line
        last_word_len = 0
        while True:
            # Get next query
            if VERBOSITY > 2:
                print("")
                print("Hit enter with no input to quit.")
            try:
                line = sys.stdin.readline().strip()
            except IOError:
                break
            debug_print("line: %s" % line, 5)

            # Extract word tokens and print those not recognized 
            line = line.lower()
            word_tokens = []
            if PHRASAL:
                word_tokens.append(line)
            else:
                word_tokens += re.split(r"\W+", line)
            for w in word_tokens:
                # Only analyze particular words once
                if w in processed:
                    continue
                processed[w] = True
                # Apply contraints specific to misspelled word
                if VERBOSITY > 1:
                    print("")
                    print("Checking word: %s" % w)
                if len(w) < MIN_WORD_LEN:
                    debug_print("Ignoring short word: %s" % w, 4)
                    continue
                if SKIP_LEADING_NUMS and (w[0] in "0123456789"):
                    # TODO: generalize to word-filter pattern
                    debug_print("Ignoring word starting with number: %s" % w, 4)
                    continue
                if CHECK_STOPLIST and w in self.is_stopword:
                    debug_print("Ignoring stopword: %s" % w, 4)
                    continue
                if CHECK_WORDLIST and w in self.is_word:
                    debug_print("Ignoring wordlist word: %s" % w, 4)
                    continue
                if CHECK_WN_ROOTS and wn_has_wordform(w):
                    debug_print("Ignoring word in wordnet: %s" % w, 4)
                    continue
                # Show spelling suggestions
                # TODO: add sanity checks for amount of suggestions and percent of replacements (esp. for default thresholds, etc.)
                try:
                    # Get suggestions, optionally with accuracy distance adjusted for word length
                    if (DYNAMIC_ACCURACY_DELTA > 0) and (len(w) != last_word_len):
                        accuracy = (len(w) - 1) / float(len(w))
                        debug_print("setting dynamic accuracy: %f" % accuracy, 2)
                        self.speller.setAccuracy(accuracy)
                        last_word_len = len(w)
                    suggestions = self.speller.suggestSimilar(w, MAX_SUGGESTIONS)
                    if VERBOSE or not index_dir:
                        print("Suggestions for %s: {%s}" % (w, ", ".join(suggestions)))
                    # Optionally find spelling replacement by checking whether
                    # any the suggestions lead to a significant increase in hits,
                    # usng case with highest document frequency. This includes 
                    # suppor for optional morphology filters.
                    # Note: this is how spelling detection currently is implemented in Juju.
                    if index_dir:
                        # TODO: have option to just find first to avoid extraneous lookup
                        replacement = None
                        num_docs = doc_freq(w)
                        freq_info = []
                        freq_info += (w, num_docs)
                        max_freq = 0
                        for alt in suggestions:
                            # Filter out known wordform variants
                            if CHECK_MORPHOLOGY and is_variant(self.derived_form, w, alt):
                                debug_print("Ignoring inflectional variants (%s, %s)" % (w, alt), 4)
                                continue
                            if CHECK_WN_MORPHOLOGY and wn_is_variant(w, alt):
                                debug_print("Ignoring wordnet inflectional variants (%s, %s)" % (w, alt), 4)
                                continue
                            if APPLY_MORPH_REWRITES and (has_morph_rewrite(self.rewrite_table, w, alt)
                                                         or has_morph_rewrite(self.rewrite_table, alt, w)):
                                debug_print("Ignoring morph-rewrite variants (%s, %s)" % (w, alt), 4)
                                continue
                            # See if new best
                            alt_freq = doc_freq(alt)
                            freq_info += (alt, alt_freq)
                            if alt_freq > max_freq:
                                replacement = alt
                                max_freq = alt_freq
                        # Check frequency constraints
                        if num_docs > MAX_MISPELLING_FREQ:
                            # TODO: put this prior to the alternatives check
                            debug_print("Ignoring word %s as frequency above threshold (%s)" % (w, MAX_MISPELLING_FREQ), 4)
                            replacement = None
                        elif (replacement and ((max_freq / float(max(num_docs, 1))) < MIN_GAIN_RATIO)):
                            # TODO: put this prior to max_freq update???
                            debug_print("Ignoring correction %s->%s as frequency ratio below threshold (%s)" % (w, replacement, MIN_GAIN_RATIO), 4)
                            replacement = None
                        # Report suggested correction
                        # Note: Include optional frequency indicator, which is formatted as
                        # token suffix to maintain whitespace tokenization [see check_replaced.py])
                        debug_print("Doc. frequencies: %s" % freq_info, 3)
                        if ADD_WORD_FREQUENCY:
                            w += "(%s)" % num_docs
                            replacement += "(%s)" % max_freq
                        print("Replacement for %s: %s" % (w, replacement))
                except:
                    if tpo.detailed_debugging():
                        tpo.debug_raise()
                    debug_print("Exception processing word '%s': %s" % (w, str(sys.exc_info())), 3)
        return

#-------------------------------------------------------------------------------

def resolve_dir_path(dir_name):
    """Resolve full path for DIR_NAME, relative to current directory unless absolute"""
    dir_path = dir_name
    if dir_name[0] not in ['/', '.']:
        dir_path = os.path.join(".", dir_name)
    tpo.debug_format("resolve_dir({d}) => {p}", 4, d=dir_name, p=dir_path)
    return dir_path
    

def main():
    """Entry point for script"""
    spell_index_dir = SPELL_INDEX_DIR
    user_dictionary = None
    index_dir = getenv_text("INDEX_DIR", None)
    field = FIELD
    reindex = False
    run_queries = False
    show_usage = (len(sys.argv) == 1)
    i = 1
    while (i < len(sys.argv)) and (sys.argv[i][0] == "-"):
        if sys.argv[i] == "--help":
            show_usage = True
        elif sys.argv[i] == "--spell-index-dir":
            i += 1
            spell_index_dir = sys.argv[i]
        elif sys.argv[i] == "--dict":
            i += 1
            user_dictionary = sys.argv[i]
        elif sys.argv[i] == "--index-dir":
            # TODO: have separate option to indicate whether to check index_dir for replacements (see run method above)
            i += 1
            index_dir = sys.argv[i]
        elif sys.argv[i] == "--index-field":
            i += 1
            field = sys.argv[i]
        elif sys.argv[i] == "--skip-query":
            run_queries = False
        elif sys.argv[i] == "--query":
            run_queries = True
        elif sys.argv[i] == "--verbose":
            global VERBOSE
            VERBOSE = True
        elif sys.argv[i] == "--re-index":
            reindex = True
        elif sys.argv[i] == "--skip-replacements":
            index_dir = None
        elif sys.argv[i] == "-":
            pass
        else:
            print("Error: unexpected argument '%s'" % sys.argv[i])
            show_usage = True
            break
        i += 1
    if show_usage:
        script = gh.basename(sys.argv[0])
        print("Usage: %s [options]" % sys.argv[0])
        print("")
        print("Options: [--spell-index-dir dir] [--dict file] [--index-dir dir] [--index-field field] [--re-index] [--query | --skip-query] [--skip-index] [--verbose] [--help]")
        print("")
        print("Example:")
        print("")
        print("%s --index-dir lucene-index-dir" % script)
        print("")
        print("echo \"tehnology\" | %s --verbose --query -" % script)
        print("")
        print("Notes:")
        print("- Use --verbose option to see all suggestions (rather than best replacement).")
        print("- The --query option is no longer the default.")
        print("- The --skip-replacements option disables checks based on index frequencies.")
        print("- The index directory is relative to current directory.")
        if VERBOSE:
            print("- Environment options:")
            print("\t%s" % tpo.formatted_environment_option_descriptions(include_all=True))
        print("")
        sys.exit()

    # Initialize the spell checking and run over stdin
    # TODO: allow the user to specify full paths
    full_spell_index_path = resolve_dir_path(spell_index_dir)
    sc = SpellCheck(full_spell_index_path)
    created_index = False
    if reindex or not os.path.exists(full_spell_index_path):
        print("Creating spelling index")
        if user_dictionary:
            full_dictionary_path = resolve_dir_path(user_dictionary)
            sc.index_dict(full_dictionary_path)
        else:
            full_index_dir = resolve_dir_path(index_dir or INDEX_DIR)
            sc.index_dir(full_index_dir, field)
        created_index = True
    if run_queries:
        sc.run(index_dir)
    else:
        if not created_index:
            tpo.print_stderr("Warning: No index created and --query option not specified")

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
