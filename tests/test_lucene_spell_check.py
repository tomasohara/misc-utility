#! /usr/bin/env python
#
# Test(s) for lucene_spell_check.py
#
# Notes:
# - This can be run as follows:
#   $ PYTHONPATH="." python tests/test_lucene_spell_check.py
#........................................................................
# Script usage for tests
#
# $ echo "tehnology" | ./lucene_spell_check.py --verbose --query -
# Suggestions for tehnology: {technology}
#
# $ echo "manager" | ./lucene_spell_check.py --verbose --query -
# Suggestions for manager: {managers, manage, managed, manger}
#

import re
import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo


class TestIt(unittest.TestCase):
    """Class for testcase definition"""
    # names for invocation via 'python -m ...'
    script_module_name = "lucene_spell_check"
    index_script_module_name = "index_table_file"

    def setUp(self):
        """Per-test initializations: indexes sample data file, disables tracing of scripts invoked via run(), and initializes temp file name (With override from environment)."""
        # Note: By default, each test gets its own temp file.
        tpo.debug_print("setUp()", 4)
        gh.disable_subcommand_tracing()
        self.temp_file = tpo.getenv_text("TEMP_FILE", tempfile.NamedTemporaryFile().name)
        self.index_dir = self.temp_file + ".index"
        self.data_file = gh.resolve_path("random10000-qiqci-query.list")
        gh.run("python  -m {self.index_script_module_name}  {self.data_file}  {self.index_dir}")
        return

    def run_script(self, options, input_text):
        """Runs the script over the DATA_FILE, passing OPTIONS"""
        tpo.debug_format("run_script({opts}, {text})", 5,
                         opts=options, text=input_text)
        self.spell_index_dir = self.temp_file + ".spell-index"
        output = gh.run("echo '{text}' | INDEX_DIR={index_dir} SPELL_INDEX_DIR={spell_index_dir} python  -m {script_module_name}  {opts} -",
                        opts=options, text=input_text,
                        index_dir=self.index_dir,
                        spell_index_dir=self.spell_index_dir,
                        script_module_name=self.script_module_name)
        # Make sure no python or bash errors
        # examples: "SyntaxError: invalid syntax", "bash: python: command not found"
        self.assertTrue(not re.search("(\S+Error:)|(no module)|(command not found)", output.lower()))
        return (output)

    def test_words(self):
        """Test a few known cases"""
        tpo.debug_print("test_words()", 4)
        self.assertTrue("managers" in self.run_script("--verbose --query", "manager"))
        self.assertTrue("technology" in self.run_script("--verbose --query", "tehnology"))
        return

    def tearDown(self):
        """Overall cleanup: deletes temporary file unless during detailed debugging"""
        tpo.debug_print("tearDownClass()", 4)
        if (tpo.debugging_level() < 4):
            gh.run("rm -rvf {self.temp_file}*")
        return

#------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

