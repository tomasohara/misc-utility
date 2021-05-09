#! /usr/bin/env python
#
# Test(s) for table_lookup.py:
# - add simple generic test (lookup phrases inserted and not find those omitted)
# - add trie-specific table
#
# Notes:
# - This can be run as follows (assuming under Bash):
#   $ PYTHONPATH="." python tests/test_table_lookup.py
# - Debugging
#   $ DEBUG_LEVEL=5 PYTHONPATH="." $PYTHON tests/test_table_lookup.py \
#     > tests/test_table_lookup.debug.log 2>&1
#

"""Unit tests for table_lookup module"""

import re
import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo


class TestIt(unittest.TestCase):
    """Class for testcase definition"""
    script_module_name = "table_lookup"  # for invocation via 'python -m ...'

    def setUp(self):
        """Per-test initializations: disables tracing of scripts invoked via 
        run(); initializes temp file name (With override from environment)."""
        # Note: By default, each test gets its own temp file.
        tpo.debug_print("setUp()", 4)
        gh.disable_subcommand_tracing()
        self.temp_file = tpo.getenv_text("TEMP_FILE", 
                                         tempfile.NamedTemporaryFile().name)
        return

    def run_script(self, options="", data_file="-", input_text="",
                   env_options=""):
        """Runs the script over the DATA_FILE, setting ENV, passing OPTIONS,
        and using INPUT_TEXT as stdin"""
        # TOD: reconcile with TestWrapper.run_script
        tpo.debug_format("run_script({opt}, {df})", 5, 
                         opt=options, df=data_file)
        gh.assertion('"' not in input_text)
        input_filename = self.temp_file + ".input.list"
        gh.write_file(input_filename, input_text)
        output = gh.run("cat {input} | {env} python  -m {mod}  {opt}  {data}",
                        input=input_filename, mod=self.script_module_name, 
                        env=env_options, opt=options, data=data_file)
        # Make sure no python or bash errors
        # exs: "SyntaxError: invalid syntax", "bash: python: command not found"
        self.assertTrue(not re.search(r"(\S+Error:)|(no module)" + 
                                      "|(command not found)",
                                      output.lower()))
        return (output)

    def check_data_file(self, env):
        """Helper to test results over a known data file, using ENV setings to
        set algoorithm, etc."""
        tpo.debug_print("test_data_file()", 4)
        #create lookup data
        data_file = "tests/sample-job-titles.list"
        table_file = self.temp_file + ".data"
        options = tpo.format("--verbose  --save-file {tf}  --skip-test", 
                             tf=table_file)
        output = self.run_script(options=options, env_options=env, 
                                 data_file=data_file)
        self.assertTrue(re.search("Saving table", output))
        # Check for known cases
        options = tpo.format("--verbose  --load-file {tf}", 
                             tf=table_file), 
        output = self.run_script(options=options, env_options=env,
                                 input_text="Clinical Psychologist" + 
                                 "\nSales Management Trainee")
        self.assertTrue(re.search("Loading table", output))
        self.assertTrue(re.search("2 .* found", output))
        # Check for unknown cases
        options = tpo.format("--verbose  --load-file {tf}", tf=table_file)
        output = self.run_script(options=options, env_options=env,
                                 input_text="Priest\nNun")
        self.assertTrue(re.search("0 .* found", output))
        return

    def test_1_trie(self):
        """Tests Trie table"""
        self.check_data_file("USE_TRIE=1")
    #
    # TODO: test trie variations (char trie and patricia trie)

    def test_2_slot_hash(self):
        """Tests Slot hash table"""
        self.check_data_file("USE_SLOT_HASH=1")
    #
    # TODO: test Phrase-slot variation

    def test_3_shelve(self):
        """Tests shelve table"""
        self.check_data_file("USE_SHELVE=1 BRUTE_FORCE=1")

    def test_4_kyoto(self):
        """Tests Kyoto cabinet table"""
        self.check_data_file("USE_KOTO=1 BRUTE_FORCE=1")

    def tearDown(self):
        """Per-test cleanup: deletes temp file unless detailed debugging"""
        tpo.debug_print("tearDown()", 4)
        if (tpo.debugging_level() < 4):
            gh.run("rm -vf {self.temp_file}")
        return

#------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

