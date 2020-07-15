#! /usr/bin/env python
#
# spaccy_ner.py: Runs spaCy named entity rercognition (NER) over text
#
# Notes:
# - Based on following:
#      https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
#      https://towardsdatascience.com/a-little-spacy-food-for-thought-easy-to-use-nlp-framework-97cbcc81f977
# - TODO: Mention limitations (e.g., prone to over-generalization).
# 
# TODO:
# - Rework in terms of spacy_nlp.py (e.g., via subclassing).
#

"""Performs named entity recognition (NER) via spaCy"""

import spacy

from main import Main
import debug
import system

# Constants (e.g., for arguments)
#
LANG_MODEL = "lang-model"
## TODO: XYZ = "xyz"

# Main class
#
class Script(Main):
    """Input processing class: runs text through spaCy NLP pipeline"""
    nlp = None
    type_prefix = ":"
    entity_delim = ", "
    entity_quote = '"'
    spacy_model = "en_core_web_lg"

    # TODO: add class constructor
    ## def __init__(self, *args, **kwargs):
    ##     debug.trace_fmtd(5, "Script.__init__({a}): keywords={kw}; self={s}",
    ##                      a=",".join(args), kw=kwargs, s=self)
    ##     super(Script, self).__init__(*args, **kwargs)
    
    def setup(self):
        """Check results of command line processing"""
        debug.trace_fmtd(5, "Script.setup(): self={s}", s=self)
        self.spacy_model = self.get_parsed_option(LANG_MODEL, self.spacy_model)
        debug.assertion(self.type_prefix != self.entity_delim)

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
            # TODO: Figure out step needed for propeer spaCy bookkeeping after a
            # new package is added to python (e.g., en_core_web_sm under site-packages).
            try:
                debug.trace(3, "Warning: Trying eval hack to load model")
                # pylint: disable=eval-used, exec-used
                exec("import " + self.spacy_model)
                debug.trace_fmt(4, "dir({m}): {d}", m=self.spacy_model, d=eval("dir(" + self.spacy_model + ")"))
                self.nlp = eval(self.spacy_model + ".load()")
            except:
                system.print_stderr("Problem with alternative load of model {m}: {exc}",
                                    m=self.spacy_model, exc=system.get_exception())
        debug.trace_object(5, self, label="Script instance")

    def process_line(self, line):
        """Processes current line from input,producing comma-separate list of entities (with type prefix)"""
        # TODO: add entity-type filter
        debug.trace_fmtd(6, "Script.process_line({l})", l=line)
        type_prefix = self.type_prefix
        entity_delim = self.entity_delim
        # Invoke NER over text
        # TODO: allow for embedded sentences
        debug.assertion(self.nlp)
        doc = self.nlp(line)

        # Format as comma-separated list of typed entities
        entity_specs = []
        for ent in doc.ents:
            ent_text = ent.text
            debug.assertion(type_prefix not in ent.label_)
            if (type_prefix in ent_text):
                ent_text = (self.entity_quote + ent_text + self.entity_quote)
            entity_specs.append(ent.label_ + type_prefix + ent_text)
        print(entity_delim.join(entity_specs))

#-------------------------------------------------------------------------------
    
if __name__ == '__main__':
    debug.trace_current_context(level=debug.QUITE_DETAILED)
    app = Script(
        description=__doc__,
        # Note: skip_input controls the line-by-line processing, which is inefficient but simple to
        # understand; in contrast, manual_input controls iterator-based input (the opposite of both).
        skip_input=False,
        manual_input=False,
        boolean_options=[],
        text_options=[(LANG_MODEL, "Language model for NER, etc.")])
    app.run()
