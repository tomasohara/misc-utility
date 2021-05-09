#! /usr/bin/env python
#
# Test(s) for ../weighted_ngrams.py
#
# Notes:
# - This can be run as follows:
#   $ PYTHONPATH=".:$PYTHONPATH" python tests/test_weighted_ngrams.py
#

"""Unit tests for weighted_ngrams module"""

import re
import unittest
from unittest_wrapper import TestWrapper

import glue_helpers as gh
import tpo_common as tpo


class TestIt(TestWrapper):
    """Class for testcase definition"""
    script_module = TestWrapper.derive_tested_module_name(__file__)
    # TODO: use data file with better weights
    query_log_file = gh.resolve_path("sample-2016-01-29.rlri.spons.data")
    title_data_file = None
    query_data_file = None

    @classmethod
    def setUpClass(cls):
        """Per-class initialization: isolate query and title data from query log file"""
        tpo.debug_format("TestIt.setupClass(): cls={c}", 5, c=cls)
        super(TestIt, cls).setUpClass()
        # Output title data file
        temp_data_base = cls.temp_base + "-" + gh.basename(cls.query_log_file)
        cls.title_data_file = temp_data_base + ".title.data"
        gh.run("cut -f7,9 {qlog} > {title_file}",
               qlog=cls.query_log_file, title_file=cls.title_data_file)
        # Output query data file
        cls.query_data_file = temp_data_base + ".query.data"
        gh.run("cut -f3,9 {qlog} > {query_file}",
               qlog=cls.query_log_file, query_file=cls.query_data_file)
        return

    def setUp(self):
        """TEMP: Fixup for setupClass invocation issue"""
        tpo.debug_format("TestIt.setUp(): self={s}", 5, s=self)
        super(TestIt, self).setUp()
        if not self.title_data_file:
            tpo.debug_print("Warning: shameless hack to invoke class setup!", 5)
            self.setupClass()
        return

    def test_ngrams(self):
        """Makes sure arbitrary ngram processing works as expected"""
        tpo.debug_print("TestIt.test_ngrams()", 4)
        ## TODO: output = self.run_script("", self.title_data_file, env_options="MIN_NGRAM=2 MAX_NGRAM=4")
        output = self.run_script("", self.query_data_file, env_options="MAX_TERMS=100 MIN_NGRAM=2 MAX_NGRAM=4")
        ## TODO: self.assertTrue(re.search(r"\nretail manager business operations\t.*\nbusiness systems analyst\t",
        self.assertTrue(re.search(r"retail manager business operations.*business systems analyst",
                                  output.strip(), re.DOTALL))
        return

    def test_bigrams(self):
        """Makes sure bigram processing works as expected"""
        tpo.debug_print("TestIt.test_bigrams()", 4)
        output = self.run_script("", self.query_data_file, env_options="MAX_TERMS=100 MIN_NGRAM=2 MAX_NGRAM=2")
        ## TODO: self.assertTrue(re.search(r"\nbilling specialist\t.*\nmanagement business\t",
        self.assertTrue(re.search(r"billing specialist.*management business\t",
                                  output.strip(), re.DOTALL))
        return

#------------------------------------------------------------------------

if __name__ == '__main__':
    tpo.trace_current_context()
    unittest.main()
