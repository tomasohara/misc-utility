#! /usr/bin/python
#
# text_processing.py: performs text processing (e.g., via NLTK)
#
# Notes:
# - function resulting caching ("memoization") is used via memodict decoration
# - environment variables (env-var) are used for some adhoc options
# - to bypass NLTK (e.g., for quick debugging), set SKIP_NLTK env-var to 0
#
#------------------------------------------------------------------------
# Miscelleneous notes
#
# Penn Tags (used by NLTK part-of-speech tagger)
# 
# CC    Coordinating conjunction
# CD    Cardinal number
# DT    Determiner
# EX    Existential there
# FW    Foreign word
# IN    Preposition or subordinating conjunction
# JJ    Adjective
# JJR   Adjective, comparative
# JJS   Adjective, superlative
# LS    List item marker
# MD    Modal
# NN    Noun, singular or mass
# NNS   Noun, plural
# NNP   Proper noun, singular
# NNPS  Proper noun, plural
# PDT   Predeterminer
# POS   Possessive ending
# PP    Personal pronoun
# PP$   Possessive pronoun ???
# PRP$  Possessive pronoun
# PRP   Personal pronoun
# RB    Adverb
# RBR   Adverb, comparative
# RBS   Adverb, superlative
# RP    Particle
# SYM   Symbol
# TO    to
# UH    Interjection
# VB    Verb, base form
# VBD   Verb, past tense
# VBG   Verb, gerund or present participle
# VBN   Verb, past participle
# VBP   Verb, non-3rd person singular present
# VBZ   Verb, 3rd person singular present
# WDT   Wh-determiner
# WP    Wh-pronoun
# WP$   Possessive wh-pronoun
# WRB   Wh-adverb
#
# See ftp://ftp.cis.upenn.edu/pub/treebank/doc/tagguide.ps.gz for more details.
#------------------------------------------------------------------------
#
# Copyright (c) 2012-2018 Thomas P. O'Hara
#

#------------------------------------------------------------------------
# Library packages

from common import *

import sys                              # system interface (e.g., command line)
import re                               # regular expressions

#------------------------------------------------------------------------
# Globals

# Skip use of NLTK and/or ENCHANT packages (using simple versions of functions)
SKIP_NLTK = getenv_boolean("SKIP_NLTK", False)
SKIP_ENCHANT = getenv_boolean("SKIP_ENCHANT", False)

# Object for spell checking via enchant
speller = None
WORD_FREQ_FILE = getenv_text("WORD_FREQ_FILE", "word.freq")

# Hash for returning most common part of speech for a word (or token)
word_POS_hash = None
WORD_POS_FREQ_FILE = getenv_text("WORD_POS_FREQ_FILE", "word-POS.freq")

# List of stopwords (e.g., high-freqency function words)
stopwords = None

#------------------------------------------------------------------------
# Optional libraries

if not SKIP_NLTK:
    import nltk				# NLP toolkit
if not SKIP_ENCHANT:
    import enchant			# spell checking

#------------------------------------------------------------------------
# Functions

def split_sentences(text):
    """Splits TEXT into sentences"""
    # EX: split_sentences("I came. I saw. I conquered!") => ["I came.", "I saw.", "I conquered!"]
    # EX: split_sentences("Dr. Watson, it's elementary. But why?") => ["Dr. Watson, it's elementary." "But why?"]
    if SKIP_NLTK:
        # Split around sentence-ending punctuation followed by space,
        # but excluding initials (TODO handle abbreviations (e.g., "mo.")
        #
        # TEST: Replace tabs with space and newlines with two spaces
        ## text = re.sub(r"\t", " ", text)
        ## text = re.sub(r"\n", "  ", text)
        # 
        # Make sure ending punctuaion followed by two spaces and preceded by one
        text = re.sub(r"([\.\!\?])\s", r" \1  ", text)
        #
        # Remove spacing added above after likely abbreviations
        text = re.sub(r"\b([A-Z][a-z]*)\s\.\s\s", r"\1. ", text)
        #
        # Split sentences by ending punctuation followed by two spaces
        # Note: uses a "positive lookbehind assertion" (i.e., (?<=...) to retain punctuation 
        sentences = re.split(r"(?<=[\.\!\?])\s\s+", text.strip())
    else:
        sentences = nltk.tokenize.sent_tokenize(text)
    return sentences


def split_word_tokens(text):
    """Splits TEXT into word tokens (i.e., words, punctuation, etc.) Note: run split_sentences first (e.g., to allow for proper handling of periods)."""
    ## EX: split_word_tokens("How now, brown cow?") => ['How', 'now', ',', 'brown', 'cow', '?']
    if SKIP_NLTK:
        tokens = [t.strip() for t in re.split("(\W+)", text) if (len(t.strip()) > 0)]
    else:
        tokens = nltk.word_tokenize(text)
    return tokens


def tag_part_of_speech(tokens):
    """Return list of part-of-speech taggings of form (token, tag) for list of TOKENS"""
    # EX: tag_part_of_speech(['How', 'now', ',', 'brown', 'cow', '?']) => [('How', 'WRB'), ('now', 'RB'), (',', ','), ('browne', 'JJ'), ('cow', 'NN'), ('?', '.')]
    if SKIP_NLTK:
        part_of_speech_taggings = [(word, get_most_common_POS(word)) for word in tokens]
    else:
        part_of_speech_taggings = nltk.pos_tag(tokens)
    return part_of_speech_taggings


def tokenize_and_tag(text):
    """Run sentence and word tokenization over text and then part-of-speecg tag it"""
    text_taggings = []
    for sentence in split_sentences(text):
        debug_print("sentence: %s" % sentence.strip(), 3)
        tokens = split_word_tokens(sentence)
        debug_print("tokens: %s" % tokens, 3)
        taggings = tag_part_of_speech(tokens)
        debug_print("taggings: %s" % taggings, 3)
        text_taggings += taggings
    return text_taggings


@memodict
def is_stopword(word):
    """Indicates whether WORD should generally be excluded from analysis (e.g., function word)"""
    global stopwords
    if (stopwords is None):
        if SKIP_NLTK:
            ## stopwords = ["the", "and", "of", "to", "a", "in", "that", "i", "it", "is", "for", "you", "was", "he", "on", "with", "as", "at", "this", "they", "be", "are", "have", "we", "but"]
            stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
        else:
            stopwords = nltk.corpus.stopwords.words('english')
        debug_print("stopwords: %s" % stopwords, 4)
    return (word.lower() in stopwords)


@memodict
def has_spelling_mistake(term):
    """Indicates whether TERM represents a spelling mistake"""
    has_mistake = False
    try:
        if SKIP_ENCHANT:
            has_mistake = term.lower() not in word_freq_hash
        else:
            has_mistake = not speller.check(term)
    except:
        debug_print("Warning: exception during spell checking of '%s': %s" % (term, str(sys.exc_info())))
    return has_mistake


# read_freq_data(filename): Reads frequency listing for words (or other keys).
# A hash table is returned from (lowercased) key to their frequency.
#
# Sample input:
#
#   # Word	Freq
#   the	179062
#   to	123567
#   is	99390
#   and	95920
#   a	76679
#
def read_freq_data(filename):
    debug_print("read_freq_data(%s)" % filename, 3)
    freq_hash = {}

    # Process each line of the file
    input_handle = open(filename,"r")
    line_num = 0;
    for line in input_handle:
        line_num += 1
        # Ignore comments
        line = line.strip()
        if (len(line) > 0) and (line[0] == '#'):
            continue

        # Extract the four fields and warn if not defined
        fields = line.split("\t")
        if (len(fields) != 2):
            debug_print("Ignoring line %d of %s: %s" % (line_num, filename, line), 3)
            continue
        (key, freq) = fields
        key = key.strip().lower()

        # Store in hash
        if key not in freq_hash:
            freq_hash[key] = freq
        else:
            debug_print("Ignoring alternative freq for key %s: %s (using %s)" 
                        % (key, freq, freq_hash[key]), 6)
        
    return (freq_hash)


# read_word_POS_data(filename): Reads frequency listing for words in particular parts of speech
# to derive dictionay of the most common part-of-speech for words (for quick-n-dirty part-of-speech
# tagging). A hash table is returned from (lowercased) words to their most common part-of-speech.
#
# Sample input:
#
#   # Token	POS	Freq
#   ,		,	379752
#   .		.	372550
#   the		DT	158317
#   to		TO	122189
#
def read_word_POS_data(filename):
    debug_print("read_word_POS_freq(%s)" % filename, 3)
    word_POS_hash = {}

    # Process each line of the file
    input_handle = open(filename,"r")
    line_num = 0;
    for line in input_handle:
        line_num += 1
        # Ignore comments
        line = line.strip()
        if (len(line) > 0) and (line[0] == '#'):
            continue

        # Extract the four fields and warn if not defined
        fields = line.split("\t")
        if (len(fields) != 3):
            debug_print("Ignoring line %d of %s: %s" % (line_num, filename, line), 3)
            continue
        (word, POS, freq) = fields
        word = word.strip().lower()

        # Store in hash
        if word not in word_POS_hash:
            word_POS_hash[word] = POS
        else:
            debug_print("Ignoring alternative POS for word %s: %s (using %s)" 
                        % (word, POS, word_POS_hash[word]), 6)
        
    return (word_POS_hash)


def get_most_common_POS(word):
    """Returns the most common part-of-speech label for WORD, defaulting to NN (noun)"""
    # EX: get_most_common_POS("can") => "MD"
    # EX: get_most_common_POS("notaword") => "NN"
    label = "NN"
    word = word.lower()
    if (word in word_POS_hash):
        label = word_POS_hash[word]
    return label

#------------------------------------------------------------------------
# Utility functions
#
# TODO: make POS optional for is_POS-type functions (and use corpus frequencies to guess)
#

def is_noun(token, POS):
    """Indicates if TOKEN is a noun, based on POS"""
    return (POS[0:2] == "NN")


def is_verb(token, POS):
    """Indicates if TOKEN is a verb, based on POS"""
    # EX: is_verb('can', 'NN') => False
    return (POS[0:2] == "VB")


def is_adverb(token, POS):
    """Indicates if TOKEN is an adverb, based on POS"""
    # EX: is_adverb('quickly', 'RB') => True
    return (POS[0:2] == "RB")


def is_adjective(token, POS):
    """Indicates if TOKEN is an adjective, based on POS"""
    # EX: is_adverb('quick', 'JJ') => True
    return (POS[0:2] == "JJ")


def is_comma(token, POS):
    """Indicates if TOKEN is a comma"""
    return ((token == ",") or (POS[0:1] == ","))


def is_quote(token, POS):
    """Indicates if TOKEN is a quotation mark"""
    ## OLD: return ((token == "'") or (token == '"'))
    # Note: this includes checks for MS Word smart quites because in training data
    # TODO: make handled properly with respect to Unicode encoding (e.g., UTF-8)
    return token in "\'\"\x91\x92\x93\x94"

def is_punct(token, POS):
    """Indicates if TOKEN is a punctuation symbol"""
    # EX: is_punct('$', '$') => True
    return (re.search("[^A-Za-z0-9]", token[0:1]) or re.search("[^A-Za-z]", POS[0:1]))

# TODO: alternative to is_punct
## def is_punctuation(token, POS):
##     """Indicates if TOKEN is a punctuation symbol"""
##     # EX: is_punct('$', '$') => True
##     # TODO: find definitive source (or use ispunct-type function)
##     punctuation_chars_regex = r"[\`\~\!\@\#\$\%\^\&\*\(\)\_\-\+\=\{\}\[\]\:\;\"\'\<\>\,\.]"
##     return (re.search(punctuation_chars_regex, token[0:1]) or (POS
## re.search("[^A-Za-z]", POS[0:1]))


#------------------------------------------------------------------------

def main():
    """
    Main routine: parse arguments and perform main processing
    TODO: revise comments
    Note: Used to avoid conflicts with globals (e.g., if this were done at end of script).
    """
    # Initialize
    debug_print("start %s: %s" % (__file__, debug_timestamp()), 3)

    # Show usage statement if no arguments or if --help specified
    if ((len(sys.argv) == 1) or (sys.argv[1] == "--help")):
        usage = """
Usage: _SCRIPT_ [--help] file
Example:

echo "My dawg has fleas" | _SCRIPT_ -

Notes:
- Intended more as a library module
- In standalone mode it runs the text processing pipeline over the file:
     sentence splitting, word tokenization, and part-of-speech tagging
- Set SKIP_NLTK environment variable to 1 to disable NLTK usage.
"""
        print_stderr(usage.replace("_SCRIPT_", __file__))
# TODO: __file__ => sys.argv[1]???

        sys.exit()

    # Run the text from each file through the pipeline
    for i in range(1, len(sys.argv)):
        # Input the entire text from the file (or stdin if - specified)
        filename = sys.argv[i]
        input_handle = open(filename, 'r') if (filename != '-') else sys.stdin

        # Analyze the text
        text = input_handle.read().strip()
        taggings = tokenize_and_tag(text)
        misspellings = [w for (w, POS) in taggings if has_spelling_mistake(w)]
        
        # Show the results
        print("text: %s" % text)
        print("taggings: %s" % taggings)
        print("misspellings: %s" % misspellings)

    # Cleanup
    debug_print("stop %s: %s" % (__file__, debug_timestamp()), 3)
    return

#------------------------------------------------------------------------
# Initialization

if SKIP_NLTK:
    word_POS_hash = read_word_POS_data(WORD_POS_FREQ_FILE)
if SKIP_ENCHANT:
    word_freq_hash = read_freq_data(WORD_FREQ_FILE)
else:
    speller = enchant.Dict("en_US")

    
if __name__ == '__main__':
    main()
