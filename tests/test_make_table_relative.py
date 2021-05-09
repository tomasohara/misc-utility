#! /usr/bin/env python
#
# Tests for make_table_relative.py
#
# Sample test performed:
#
#   $ cat tests/sample-data-table.data 
#   Category	Count
#   cat 1	10
#   cat 2	20
#
#   $ make_table_relative.py --cat-regex 'cat (\S+)' --ref-cat 2  tests/sample-data-table.data 
#   Category	Count
#   cat 1	10.0 (-0.5)
#   cat 2	20.0 (*)
#------------------------------------------------------------------------
# Usage for script being tested:
# 
# 
# $ python -m make_table_relative --help
# usage: make_table_relative.py [-h] [--label-field LABEL_FIELD]
#                               [--cat-regex CAT_REGEX] [--delim DELIM]
#                               [--ignore-cats IGNORE_CATS] [--ref-cat REF_CAT]
#                               [--header] [--no-header] [--just-diff]
#                               [filename]
# 
# Converts valus in table to be relative to reference row(s)
# 
# positional arguments:
#   filename              Input filename
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   --label-field LABEL_FIELD
#                         Column number for row label
#   --cat-regex CAT_REGEX
#                         Pattern for deriving category from label field
#   --delim DELIM         Delimiter for fields
#   --ignore-cats IGNORE_CATS
#                         String list of categories to ignore
#   --ref-cat REF_CAT     Reference category for deriving relative differences
#   --header              Table includes header row (default)
#   --no-header           Table doesn't have header row
#   --just-diff           Only displays the relative differences
#------------------------------------------------------------------------
# Notes:
# - This can be run as follows:
#   $ PYTHONPATH="." python tests/test_make_table_relative.py
# - It uses output from the master A/B test script under the query-log-analysis
#    repository(see get_logistic_regression_inputs/master_ab_test.py). So this 
#    should be updated if major changes are made to A/B test output.
#------------------------------------------------------------------------
# TODO:
# - Test both directions for relative performance.
# - Test implicit reference category support (i.e., first one encountered).
#

import re
import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo

class TestIt(unittest.TestCase):
    """Class for testcase definition"""
    script_module_name = "make_table_relative"

    def setUp(self):
        """Per-test initializations, including disabling tracing of scripts via run()"""
        # Note: TEMP_FILE env. var. overrides random file name to simplify debugging
        tpo.debug_print("setUp()", 4)
        gh.disable_subcommand_tracing()
        self.temp_file = tpo.getenv_text("TEMP_FILE", tempfile.NamedTemporaryFile().name)
        return

    def run_script(self, options, data_file):
        """Runs the script over the sample DATA_FILE, passing OPTIONS"""
        tpo.debug_format("run_master({options}, {data_file})", 5)
        data_file = gh.resolve_path(data_file)
        output = gh.run("python  -m {self.script_module_name}  {options}  {data_file}")
        self.assertTrue(not re.search("(error:)|(no module)", output.lower()))
        return (output)

    def test_dummy_data(self):
        """Test mocked-up data with known proportions"""
        tpo.debug_print("test_sample_data()", 4)
        table = [
            "Category	Count",
            "cat 1	10",
            "cat 2	20",
            ]
        gh.write_lines(self.temp_file, table)
        output = self.run_script("--cat-regex 'cat (\S+)'  --ref-cat 2", self.temp_file)
        # Category 1's 10 value is 50% worse relative to category 2's 20.
        self.assertTrue(re.search("cat 1.*\s10.*\(-0.5\)", output))
        self.assertTrue(re.search("cat 2.*\s20.*\(\*\)", output))
        return

    def test_sample_data(self):
        """Test sample data used in master_ab_test script test"""
        tpo.debug_print("test_sample_data()", 4)
        # Extract plain table from input HTML and then convert table include relative difference.
        data_file = gh.resolve_path("sample-consolidated-query-log.html")
        gh.run("python -m extract_html_table_data  {data_file}  >|  {self.temp_file}")
        output = self.run_script("--cat-regex '\d\d\d\d\d[A-Z]?'  --ref-cat 12906", 
                                 self.temp_file)
        # Verify that algorithm 12396B's users value of 4 is 33% higher relative to algorithm 12906's 3.
        self.assertTrue(re.search("total.*12396B.*\s4.*\(0.33.*\)", output))
        self.assertTrue(re.search("total.*12906.*\s3.*\(\*\)", output))
        return
        
if __name__ == '__main__':
    unittest.main()
