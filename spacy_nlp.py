#! /usr/bin/env python
#
# spaccy_ner.py: Runs spaCy for a variety of natural lanuage processing (NLP)
# tasks.
#
# Notes:
# - This is a generalization of spacy_ner.py, which was just for named entity recognition (NER).
# - Unforunately spaCy's document omits important detail that sentiment analysis is not built in!
# - To compensate, sentiment analyzer is based on vader:
#      https://medium.com/swlh/simple-sentiment-analysis-for-nlp-beginners-and-everyone-else-using-vader-and-textblob-728da3dbe33d
# TODO:
# - Add support for parsing.
# - Integrate state-of-the-art sentiment analysis.
# 

"""Performs various natural lanuage processing (NLP) tasks via spaCy"""

# Standard packages
import re

# External packages
import spacy

# Local packages
from main import Main
import debug
import system

# Constants (e.g., for arguments)
#
VERBOSE = "verbose"
RUN_NER = "run-ner"
ANALYZE_SENTIMENT = "analyze-sentiment"
## TODO: RUN_POS_TAGGER = "pos-tagging"
## TODO: RUN_TEXT_CATEGORIZER = "text-categorization"
LANG_MODEL = "lang-model"
## TODO: TASK_MODEL = "task-model"
USE_SCI_SPACY = "use-scispacy"

#...............................................................................

# HACK: placeholder for optional sentiment module
SentimentIntensityAnalyzer = None

class SentimentAnalyzer(object):
    """Class for analyzing sentiment of sentences or words"""
    # Uses VADER: Valence Aware Dictionary and sEntiment Reasoner
    # Examples:
    # bad:  {'neg': 1.0, 'neu': 0.0, 'pos': 0.0, 'compound': -0.5423}
    # good: {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.4404}
    analyzer = None
    
    def __init__(self):
        """Class constructor"""
        # Make sure vader is initialized
        global SentimentIntensityAnalyzer
        if not SentimentIntensityAnalyzer:
            debug.trace(2, "Warning: shameless hack for dynamic loading of vader")
            # pylint: disable=import-outside-toplevel
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as sia
            SentimentIntensityAnalyzer = sia
        self.analyzer = SentimentIntensityAnalyzer()

    def get_score(self, text):
        """Return sentiment score for word(s) in text"""
        debug.trace_fmt(5, "get_sentiment({t})", t=text)
        all_scores = self.analyzer.polarity_scores(text)
        debug.trace_fmt(6, "scores={all}", all=all_scores)
        score = all_scores.get("compound", 0)
        debug.trace_fmt(4, "get_sentiment({t}) => {s}", t=text, s=score)
        return score

#...............................................................................

# Main class
#
class Script(Main):
    """Input processing class"""
    nlp = None
    type_prefix = ":"
    entity_delim = ", "
    entity_quote = '"'
    spacy_model = "en_core_web_lg"
    analyze_sentiment = False
    run_ner = False
    show_representation = False
    verbose = False
    doc = None
    sentiment_analyzer = None
    use_sci_spacy = False

    def setup(self):
        """Check results of command line processing"""
        debug.trace_fmtd(5, "Script.setup(): self={s}", s=self)
        self.spacy_model = self.get_parsed_option(LANG_MODEL, self.spacy_model)
        self.analyze_sentiment = self.get_parsed_option(ANALYZE_SENTIMENT, self.analyze_sentiment)
        self.run_ner = self.get_parsed_option(RUN_NER, self.run_ner)
        self.verbose = self.get_parsed_option(VERBOSE, self.verbose)
        default_use_sci_spacy = re.search(r"\bsci\b", self.spacy_model)
        self.use_sci_spacy = self.get_parsed_option(USE_SCI_SPACY, default_use_sci_spacy)
        ## TODO: self.do_x = self.get_parsed_option(X, self.do_x)
        do_specific_task = (self.run_ner or self.analyze_sentiment)
        ## TODO: self.show_representation = self.verbose or (not do_specific_task)
        self.show_representation = (not do_specific_task)
        self.doc = None

        # HACK: Load in external module for sentiment analysis (gotta hate spaCy!)
        if self.analyze_sentiment:
            self.sentiment_analyzer = SentimentAnalyzer()

        # Load SpaCy language model (normally large model for English)
        debug.trace_fmt(4, "loading SpaCy model {m}", m=self.spacy_model)
        try:
            self.nlp = spacy.load(self.spacy_model)
        except:
            system.print_stderr("Problem loading model {m} via spacy: {exc}",
                                m=self.spacy_model, exc=system.get_exception())
            # If model package not properly installed, the model can be
            # created by the following workaround (n.b., uses specific model for clarity);
            #    nlp = spacy.load("en_core_web_sm")
            #        =>
            #    import en_core_web_sm
            #    nlp = en_core_web_sm.load()
            # TODO: Figure out step needed for propeer spaCy bookkeeping after 
            # new package is added to python (e.g., en_core_web_sm under site-packages).
            try:
                debug.trace(3, "Warning: Trying eval hack to load model")
                # pylint: disable=eval-used, exec-used
                if self.use_sci_spacy:
                    SCISPACY = "scispacy"
                    exec("import {p}".format(p=SCISPACY))
                    debug.trace_fmt(5, "dir({p}): {d}", p=SCISPACY, d=eval("dir(" + SCISPACY + ")"))
                exec("import " + self.spacy_model)
                debug.trace_fmt(5, "dir({m}): {d}", m=self.spacy_model, d=eval("dir(" + self.spacy_model + ")"))
                self.nlp = eval(self.spacy_model + ".load()")
            except:
                system.print_stderr("Problem with alternative load of model {m}: {exc}",
                                    m=self.spacy_model, exc=system.get_exception())
        # Sanity checks
        debug.assertion(self.nlp)
        debug.assertion(self.type_prefix != self.entity_delim)
                
        debug.trace_object(6, self, label="Script instance")


    def get_entity_spec(self):
        """Return the named entities tagged in the input as string list of typed entities"""
        ## EX: "PERSON:Trump, ORG:the White House"
        ## TODO: return regular list
        debug.trace_fmt(4, "get_entity_spec(): self={s}", s=self)

        # Collect list of entities (e.g., ["PERSON:Elon Musk,", "ORG:SEC"])
        entity_specs = []
        for ent in self.doc.ents:
            ent_text = ent.text
            debug.assertion(self.type_prefix not in ent.label_)
            if (self.type_prefix in ent_text):
                ent_text = (self.entity_quote + ent_text + self.entity_quote)
            entity_specs.append(ent.label_ + self.type_prefix + ent_text)

        # Convert to string
        entity_spec = self.entity_delim.join(entity_specs)
        debug.trace_fmt(5, "get_entity_spec() => {r}", r=entity_spec)
        return entity_spec

    def get_sentiment_score(self):
        """Returns the overall sentiment score associated with the text"""
        debug.trace_fmt(4, "get_sentiment_score(): self={s}", s=self)
        # TODO: show word-level sentiment scores, highlighting cases much
        # different from the overall score
        ## BAD: score = self.doc.sentiment
        score = self.sentiment_analyzer.get_score(system.to_string(self.doc))
        debug.trace_fmt(4, "get_sentiment_score() => {r}", r=score)
        return score
    
    def process_line(self, line):
        """Processes current line from input,producing comma-separate list of entities (with type prefix)"""
        # TODO: add entity-type filter
        debug.trace_fmtd(6, "Script.process_line({l})", l=line)

        # Analyze the text in the line
        # TODO: allow for embedded sentences
        self.doc = self.nlp(line)
        debug.trace_object(5, self.doc, "doc")
        if self.verbose:
            print("input: " + line)
        # Show synopsis of word token representations
        # TODO: have options for specifying attributes to show
        if self.show_representation:
            # Gather the word (lexeme) information
            # Note: Based on vocab/lexemes section of https://spacy.io/usage/spacy-101
            attributes = ["text", "is_oov", "is_stop", "sentiment"]
            all_info = []
            for word in self.doc:
                lexeme = self.doc.vocab[word.text]
                info = [getattr(lexeme, a, "") for a in attributes]
                all_info.append("\t".join([system.to_string(v) for v in info]))
            # Output the synopsis
            prefix = ("\t".join(attributes) + "\n")
            doc_repr = "\n".join(all_info)
            print(prefix + doc_repr)

        # Optionally, invoke NER
        if self.run_ner:
            prefix = "entities: " if self.verbose else ""
            print(prefix + self.get_entity_spec())
        # Optionally, do sentiment analysis
        if self.analyze_sentiment:
            prefix = "sentiment: " if self.verbose else ""
            print(prefix + system.to_string(self.get_sentiment_score()))
        
#-------------------------------------------------------------------------------
    
if __name__ == '__main__':
    debug.trace_current_context(level=debug.QUITE_DETAILED)
    app = Script(
        description=__doc__,
        # Note: skip_input controls the line-by-line processing, which is inefficient but simple to
        # understand; in contrast, manual_input controls iterator-based input (the opposite of both).
        skip_input=False,
        manual_input=False,
        boolean_options=[RUN_NER, ANALYZE_SENTIMENT, VERBOSE],
        text_options=[(LANG_MODEL, "Language model for NLP")])
    app.run()
