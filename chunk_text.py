#! /usr/bin/env python
#
# Runs text through chunker, grouping sets of words based on related tags,
# based on code from text extraction chapter of book on NLTK:
#    http://nltk.googlecode.com/svn/trunk/doc/book/ch07.html
# Note: Chunks do no embed, so a NP inside a PP will be treated as two chunks.
#
# Sample input:
#    I spoke briefly with the young man.
#
# Sample output:
#    (S 
#       (VP I spoke briefly)
#       with
#       (NP the young man))
#
# Notes:
# - URLs below for NLTK document assume NLTKDOC stands for the following prefix:
#       nltk.googlecode.com/svn/trunk/doc
#  For example, the URL above for chapter 7 would be given as NLTKDOC/book/ch07.html.
# - __debug__ is on by default and turned off by the -O switch (optimize).
# - When loading the first time, the chunker module will spend a bit of time training
#   the models.
#
# TODO:
# - Have option to take part-of-speech tagged input.
# - Use debug module for conditional tracing.
#

"""Run the text through NLP chunker to group together related parts-of-speech"""

# for Debugging purpose
## DEBUG: from datetime import datetime
## DEBUG: import inspect

# load required libraries
import sys
## DEBUG: if __debug__: print >> sys.stderr, "start:", str(datetime.now())
## OLD: import re
import nltk
from nltk.corpus import conll2000
import debug

#.......................................................................
# Supporting code

# "unigram" chunker form NLTK/book/ch07.html. This trains the chunker over
# tagged data from the CoNLL-2000 competition.
# note: For chunker interface, see NLTKDOC/api/nltk.chunk.api.ChunkParserI-class.html.
# TODO2: use bigram-chunker
# TODO3: integrate changes from my-chunker.py

class UnigramChunker(nltk.ChunkParserI):

    """Chunker based on unigram sequences (i.e., no context)"""
    
    def __init__(self, train_sents):
        """Class constructor for taking tagged chunked text and deriving classifier."""
        # Note: See NLTKDOC/api/nltk.tag.sequential.UnigramTagger-class.html.
        ## DEBUG: if __debug__: print >> sys.stderr, "in UnigramChunker:__init__; train_sents =", train_sents
        train_data = [[(t, c) for (_w, t, c) in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        ## DEBUG: if __debug__: print >> sys.stderr, "train_data =", train_data
        self.tagger = nltk.UnigramTagger(train_data)

    def parse(self, tokens):
        """Chunk part-of-speech tagged TOKENS and return as parse tree."""
        # ex: parse([('How', 'WRB'), ('now', 'RB'), ('brown', 'VBN'), ('cow', 'NN'), ('?', '.')]) => (S How/WRB now/RB (VP brown/VBN) (NP cow/NN) ?/.
        # TODO3: rename to something like chunk_tagged_text
        # See NLTKDOC/api/nltk.chunk.util-module.html.
        ## DEBUG: if __debug__: print >> sys.stderr, "in UnigramChunker:parse; sent =", sent
        pos_tags = [pos for (_word, pos) in tokens]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        ## DEBUG: if __debug__: print >> sys.stderr, "tagged_pos_tags =", tagged_pos_tags
        chunktags = [chunktag for (_pos, chunktag) in tagged_pos_tags]
        ## DEBUG: if __debug__: print >> sys.stderr, "chunktags =", chunktags
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag)
                     in zip(tokens, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

    def chunk(self, sentence):
        """Chunk part-of-speech tagged sentence and return list of chunk tags"""
        # ex: chunk([('How', 'WRB'), ('now', 'RB'), ('brown', 'VBN'), ('cow', 'NN'), ('?', '.')]) => ['O', 'O', 'I-VP', 'I-NP', 'O']"""
        ## DEBUG: if __debug__: print >> sys.stderr, "in UnigramChunker:chunk; sentence =", sentence
        pos_tags = [pos for (_word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        return chunktags

    def grammar(self):
        """Returns the grammar used by this parser."""
        debug.assertion(False)
        return ""

    
#........................................................................
# TODO: encapslate in function

def main():
    """Entry point for script"""

    # Initialze
    parse_tree = False				# generate parse-tree ouput
    
    # Parse command-line, showing usage statement if requested.
    ## DEBUG: if __debug__: print >> sys.stderr, "argv =", sys.argv
    if (len(sys.argv) > 1):
        if (sys.argv[1] == "--help"):
            print >> sys.stderr, "Usage: ", sys.argv[0], " [--parse-tree] [--help]"
            sys.exit()
        if (sys.argv[1] == "--parse-tree"):
            parse_tree = True
        
    # Train chunker
    # Note: this uses tagged data from CoNLL chunker competition from 2000
    # TODO: cache the resulting classifier in a file; also, use BigramChunker
    ## DEBUG: if __debug__: print >> sys.stderr, "Training unigram tagger"
    training_sentences = conll2000.chunked_sents('train.txt')
    ## DEBUG: if __debug__: print >> sys.stderr, "at line ", inspect.currentframe().f_lineno, ": ", str(datetime.now())
    unigram_chunker = UnigramChunker(training_sentences)
    
    # Process input
    # - Read in list of words and parts of speech
    # - Derive lemma for each
    # - Write out lemmatization informaiont
    while True:
        # Get the next line
        text = sys.stdin.readline()
        if not text:
            break
        ## DEBUG: if __debug__: print >> sys.stderr, "Text: ", text
    
        # Tokenize and part-of-speech tag the text
        # Notes:
        # - word_tokenize Use NLTK's currently recommended word tokenizer to tokenize words in the given sentence. Currently, this uses TreebankWordTokenizer. This tokenizer should be fed a single sentence at a time.
        #   [NLTKDOC/api/nltk.tokenize-module.html#word_tokenize]
        # - A word tokenizer that tokenizes sentences using the conventions used by the Penn Treebank.  Contractions, such as "can't", are split in to two tokens.
        #   [NLTKDOC/api/nltk.tokenize.treebank.TreebankWordTokenizer-class.html]
        tokenized_text = nltk.word_tokenize(text)
        ## DEBUG: if __debug__: print >> sys.stderr, "tokenized_text =", tokenized_text
        tagged_text = nltk.pos_tag(tokenized_text)
        ## DEBUG: if __debug__: print >> sys.stderr, "tagged_text =", tagged_text
    
        # Chuck the tagged text and print the result
        chunked_text = (unigram_chunker.parse(tagged_text) if parse_tree else unigram_chunker.chunk(tagged_text))
        print chunked_text
    
    # TODO: Close up shop
    
    ## DEBUG: if __debug__: print >> sys.stderr, "stop:", str(datetime.now())

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
