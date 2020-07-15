# tpo_wordnet.py: Tom's module for WordNet access using NLTK WordNet package
#
# notes:
# - This is one of my existing modules used just for prototyping. I'll
# rewrite anything intended for long-term use (e.g., production).
#

from tpo_common import *
debug_print("wordnet.py: " + debug_timestamp(), level=3)

import re
import nltk
debug_print("after importing NLTK: " + debug_timestamp(), level=3)
from nltk.corpus import wordnet as wn
debug_print("after importing wordnet support: " + debug_timestamp(), level=3)

import system

#------------------------------------------------------------------------

# Labels used to distinguish words with different parts of speech: 'n', 'v', 'a', 'b', and '?'
UNKNOWN_PART_OF_SPEECH = "?"
WORDNET_PARTS_OF_SPEECH = [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV, UNKNOWN_PART_OF_SPEECH]


# get_part_of_speech_prefix([part_of_speech=None]): Get prefix used for words with given PART_OF_SPEECH.
# EX: get_part_of_speech_prefix("n") => "n-"
# EX: get_part_of_speech_prefix() => ""
#
def get_part_of_speech_prefix(part_of_speech=None):
    prefix = ""
    if part_of_speech and (len(part_of_speech) > 0):
        prefix = part_of_speech + ":"
    debug_print("get_part_of_speech_prefix(%s) => %s" % (str(part_of_speech), prefix), level=9)
    return prefix

# get_word_spec(word, part_of_speech=None): Return wordform specification given WORD and optional PART_OF_SPEECH
# EX: word_spec("dog") => "dog"
# EX: word_spec("dog", WN.noun) => "n:dog"
def get_word_spec(word, part_of_speech=None):
    prefix = get_part_of_speech_prefix(part_of_speech)
    word_spec = prefix + word
    debug_print("get_word_spec('%s', %s) => %s" % (word, str(part_of_speech), word_spec), level=7)
    return word_spec

# parse_wordform(wordform): Return part of speech and word proper (as a tuple).
# EX: parse_wordform("dog") => (None, "dog")
# EX: parse_wordform("n:dog") => (wn.NOUN, "dog")
#
def parse_wordform(wordform):
    part_of_speech = None
    word = wordform
    match = re.search(r"(\w+):(\w+)", word)
    if match:
        part_of_speech = match.group(1)
        word = match.group(2)
        assert(part_of_speech in WORDNET_PARTS_OF_SPEECH)
    debug_print("parse_wordform(%s) => %s" % (wordform, str((part_of_speech, word))), level=7)
    return (part_of_speech, word)

# get_part_of_speech(tag, default='?'): Returns WordNet part-of-speech label for Treebank-style TAG.
# Note: WordNet only includes nouns, verbs, adjectives and adverbs
# EX: get_wordnet_part_of_speech("NNS") => wn.NOUN
#
def get_part_of_speech(tag, default=UNKNOWN_PART_OF_SPEECH):
    part_of_speech = default
    if (re.search("^NN", tag)):
        part_of_speech = wn.NOUN
    elif (re.search("^VV", tag)):
        part_of_speech = wn.VERB
    elif (re.search("^JJ", tag)):
        part_of_speech = wn.ADJ
    elif (re.search("^RB", tag)):
        part_of_speech = wn.VERB
    debug_print("get_part_of_speech(%s) => %s" % (tag, str(part_of_speech)), level=7)
    return part_of_speech

# get_root_word(wordform, [part_of_speech=None]): Apply simple morpholigy to derive root for WORDFORM.
# EX: get_root_word("written") => "write"
#
def get_root_word(word, part_of_speech=None):
    root = wn.morphy(word, part_of_speech)
    debug_print("get_root_word(%s, %s) => %s" % (word, str(part_of_speech), root), level=8)
    return wn.morphy(word)

# get_synset(synset_spec): Returns synset given "<WORD>.<POS>.<SENSE>" specification.
# EX: re.search("person.*practice.*law", get_synset("laywer.n.01").definition())
#
def get_synset(synset_spec):
    synset = None
    assert(re.match(r"^\w+\.\w\.\d+$", synset_spec))
    try:
        synset = wn.synset(synset_spec)
    except:
        debug_raise()
        print_stderr("Exception in get_synset: " + str(sys.exc_info()))
    debug_print("get_synset(%s) => %s" % (str(synset_spec), str(synset)), level=7)
    return synset

# get_lemma_word(lemma, [prefix=""]): Returns word for LEMMA with optional PREFIX.
# Note: Underscores are replaced by spaces.
# EX: get_lemma_word(Lemma('lawyer.n.01.attorney')) => attorney
#
def get_lemma_word(lemma, prefix=""):
    assert(re.search("wordnet.lemma", str(type(lemma)).lower()))
    word = ""
    try:
        word = prefix + lemma.name.replace("_", " ")
    except:
        debug_raise()
        print_stderr("Exception in get_lemma_word: " + str(sys.exc_info()))
    debug_print("get_lemma_word(%s, '%s') => %s" % (str(lemma), prefix, str(word)), level=8)
    return (word)

# get_synset_words(synset, [part_of_speech_prefi=""]): Returns the words used to refer to SYNSET,
# using optional part-of-speech PREFIX.
# EX: get_synset_words(wn.synset("laywer.n.01", "n:") => ["n:lawyer", "n:attorney"]
#
def get_synset_words(synset, prefix=""):
    words = []
    assert(re.search("wordnet.synset", str(type(synset)).lower()))
    try:
        words = [get_lemma_word(lemma, prefix) for lemma in synset.lemmas]
    except:
        debug_raise()
        print_stderr("Exception in get_synset_words: " + str(sys.exc_info()))
    debug_print("get_synset_words(%s, '%s') => %s" % (str(synset), prefix, str(words)), level=7)
    return words

# get_synonyms(wordform): Returns list of synonyms for WORDFORM based on WordNet.
# If the input word has a part-of-speech prefix (e.g., "v:can"), so will the resulting words.
# EX: get_synonyms("attorney") => ["lawyer"]
# EX: get_synonyms("n:attorney") => ["n:lawyer"]
# EX: ("v:fire" in get_synonyms("v:can") and "v:fire" not in get_synonyms("n:can"))
#
def get_synonyms(wordform):
    synonyms = []
    
    # See if optional part-of-speech indicator present
    (part_of_speech, word) = parse_wordform(wordform)

    # Check each of the synsets for word for the lemma's (i.e., dictionary base word)
    try:
        word_base = get_root_word(word, part_of_speech)
        for synset in wn.synsets(word, part_of_speech):
            words = [w for w in get_synset_words(synset) if (w != word_base)]
            word_forms = [get_word_spec(w, part_of_speech) for w in words]
            synonyms += word_forms
    except:
        debug_raise()
        print_stderr("Exception in get_synonyms: " + str(sys.exc_info()))
    debug_print("get_synonyms(%s) => %s" % (wordform, str(synonyms)), level=7)
    return (synonyms)

# get_synset_hypernyms(synset, [max_link=1], [processed=None]): Returns anector terms for SYNSET using at most MAX_DIST link, using PROCESSED to check for cycles.
# EX: get_synset_hypernyms(Synset('lawyer.n.01')) => [Synset('professional.n.01')]
def get_synset_hypernyms(synset, max_dist=1, processed=None, indent=""):
    debug_print("%sget_synset_hypernyms%s" % (indent, str((synset, max_dist, "_,_"))), level=7)
    assert(re.search("wordnet.synset", str(type(synset)).lower()))
    hypernyms = []

    # Get the immediate hypernyms
    try:
        hypernyms = synset.hypernyms()
    except:
        debug_raise()
        print_stderr("Exception in get_synset_hypernyms: " + str(sys.exc_info()))

    # If more links desired, recursively get the ancestors
    if (max_dist > 1):
        if not processed:
            processed = dict()
        all_hypernyms = hypernyms
        for hypernym in hypernyms:
            if processed.has_key(hypernym):
                debug_print("Skipping already processed hypernym: " + str(hypernym), level=8)
                continue;
            processed[hypernym] = True
            all_hypernyms += get_synset_hypernyms(hypernym, (max_dist - 1), processed, (indent + "\t"))
    debug_print("%sget_synset_hypernyms(%s,_,_) => %s" % (indent, str(synset), str(hypernyms)), level=7)
    return hypernyms

# get_hypernym_terms(wordform, [max_dist=1]): Returns ancestor terms for WORDFORM using at most MAX_DIST links.
# EX: get_hypernym_terms("n:attorny") => "n:professional"
#
def get_hypernym_terms(word, max_dist=1):
    hypernym_terms = []
    
    # See if optional part-of-speech indicator present
    (part_of_speech, word) = parse_wordform(word)
    part_of_speech_prefix = get_part_of_speech_prefix(part_of_speech)

    # Extract terms from each of the hypernym synsets
    try:
        for synset in wn.synsets(word, part_of_speech):
            for hypernym in get_synset_hypernyms(synset, max_dist):
                hypernym_terms += get_synset_words(hypernym, part_of_speech_prefix)
    except:
        debug_raise()
        print_stderr("Exception in get_hypernym_terms: " + str(sys.exc_info()))
    debug_print("get_hypernym_terms(%s) => %s" % (word, str(hypernym_terms)), level=7)
    return (hypernym_terms)

#------------------------------------------------------------------------

# Warn if invoked standalone
#
if __name__ == '__main__':
    if not __debug__:
        print_stderr("Warning: wordnet.py is not intended to be run standalone")
    else:
        debug_print("n:lawyer wordform: " + str(parse_wordform("n:lawyer")))
        debug_print("base for written: " + str(get_root_word("written")))
        debug_print("Synset for lawyer.n.01: " + str(get_synset("lawyer.n.01")))
        debug_print("Synonyms of lawyer: " + str(get_synonyms("n:lawyer")))
        debug_print("Immediate hypernym terms of lawyer: " + str(get_hypernym_terms("n:lawyer")))
        debug_print("All hypernym terms of lawyer: " + str(get_hypernym_terms("n:lawyer", max_dist=system.MAX_INT)))
    debug_print("end: " + debug_timestamp(), level=3)
