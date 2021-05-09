#! /usr/bin/env python
#
# Test(s) for ../kenlm_example.py
#
# Notes:
# - Automates the following steps:
#   $ {
#      PATH="$HOME/programs/kenlm/bin:$PATH"
#      train_language_model.py keyword-query-count.query.list
#      echo $'administrative assistant\nprogramming assistant\n' | LM=keyword-query-count.query.list.3gram.arpa SKIP_SENT_TAGS=1 kenlm_example.py -
#   }
#   =>
#   ...
#   -2.17061209679 2 : "administrative assistant"
#   ...
#   -3.75845742226 2 : "programming assistant"
#
# - The KenLM binaries directory should be in the path (as above); and, the parent directory should
#   be in the python path. So this can be run as follows:
#   $ PATH="$HOME/programs/kenlm/bin:$PATH" PYTHONPATH=".:$PYTHONPATH" python -u tests/test_kenlm_example.py
#
# ........................................................................
# TODO:
# - Use smaller input data file (to speed up indexing).
# - Remove output file unless debugging or REUSE_OUTPUT set.
#

"""Unit tests for kenlm_example module"""

import math
import re
import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo

SAMPLE_DATA_FILE = "random-query.list"
COMMON_BIGRAM = "administrative assistant"
UNCOMMON_BIGRAM = "manager assistant"
REUSE_OUTPUT = tpo.getenv_boolean("REUSE_OUTPUT", tpo.detailed_debugging())

class TestIt(unittest.TestCase):
    """Class for testcase definition"""
    script_module_name = "kenlm_example"  # name for invocation via 'python -m ...'

    def setUp(self):
        """Per-test initializations, including disabling tracing of scripts invoked via run()"""
        tpo.debug_print("setUp()", 5)
        self.temp_file = tpo.getenv_text("TEMP_FILE", tempfile.NamedTemporaryFile().name)
        self.model_file = None
        gh.disable_subcommand_tracing()
        self.check_program_usage()
        self.check_module_accessibility()
        return

    def check_program_usage(self):
        """Makes sure program accessible and expected options supported"""
        # Show usage statement
        tpo.debug_print("check_program_usage()", 4)
        usage = gh.run("lmplz").lower()
        # Make sure supporting program accessible and that usage statement is as expected
        assert (not re.search("not found", usage)), "*** Requires lmplz from kenlm toolkit ***"
        assert re.search("ken.*heafield", usage)
        assert re.search("-o.*order.*model", usage)
        assert re.search("--memory", usage)
        return

    def check_module_accessibility(self):
        """Makes sure python module installed"""
        tpo.debug_print("check_module_accessibility()", 5)
        import_output = gh.run("python -c 'import kenlm'")
        assert not re.search("ImportError", import_output)
        return

    def setup_model(self):
        """Create common model file used in tests"""
        tpo.debug_print("setup_model()", 5)
        input_path = self.temp_file + ".txt"
        self.model_file = self.temp_file + ".3gram.arpa"
        log_file = tpo.format("{self.temp_file}.{self.script_module_name}.log")

        # Create the language model
        sample_data_path = gh.resolve_path(SAMPLE_DATA_FILE)
        gh.copy_file(sample_data_path, input_path)
        run_training = not (REUSE_OUTPUT and (gh.non_empty_file(self.model_file)))
        if run_training:
            gh.run("python -m train_language_model --interpolate-unigrams {input_path} >| {log_file} 2>&1")
            self.assertTrue(gh.non_empty_file(self.model_file))
        else:
            tpo.debug_format("Warning: Using existing model file {self.model_file}")
        return

    def run_query(self, query):
        """Runs QUERY though language model, returning score in range [0, 1]"""
        # Note: keep this up to 
        info = gh.run("echo '{query}' | LM={self.model_file} python -m {self.script_module_name} -")
        count = tpo.safe_float(gh.extract_match(tpo.format(r"sentence: {query}\s*model score: (\S+)"), [info]))
        score = math.exp(count)
        tpo.debug_format("run_query({query}) => {score}", 4)
        return score

    def test_bigram(self):
        """Ensures that common bigram scored higher than uncommon one"""
        tpo.debug_print("test_ngram()", 4)
        self.setup_model()

        # Make sure the common bigram rated higher
        common_score = self.run_query(COMMON_BIGRAM)
        uncommon_score = self.run_query(UNCOMMON_BIGRAM)
        self.assertTrue(common_score > uncommon_score)
        return

    def test_ngram(self):
        """Ensures that a ngram from constituent bigrams ranks higher than ngrams over unigrams; also checks for ngrams from unseen words ranks lowers than those over seen words."""
        tpo.debug_print("test_ngram()", 4)
        self.setup_model()
        gh.assertion(gh.non_empty_file(self.model_file))
        # Check scores for ngrams built over over known vs. unknown bigrams
        seen_bigrams_score = self.run_query("delivery driver needed urgent care")
        unseen_bigrams_score = self.run_query("urgent needed delivery care driver")
        self.assertTrue(unseen_bigrams_score < seen_bigrams_score)
        # Check for unseen words
        unseen_words_score = self.run_query("w0rd1 w0rd2 w0rd3 w0rd4 w0rd5")
        self.assertTrue(unseen_words_score < unseen_bigrams_score)
        return

    def tearDown(self):
        """Per-test cleanup: deletes temporary files unless during detailed debugging"""
        tpo.debug_print("tearDown()", 5)
        if not tpo.detailed_debugging():
            gh.run("rm -vf {self.temp_file}*")
        return

if __name__ == '__main__':
    unittest.main()
