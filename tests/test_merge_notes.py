#! /usr/bin/env python
#
# Test(s) for ../merge_notes.py
#
# Notes:
# - Fill out TODO's below. Use numbered tests to order (e.g., test_1_usage).
# - This can be run as follows:
#   $ PYTHONPATH=".:$PYTHONPATH" python tests/test_merge_notes.py
#
#-------------------------------------------------------------------------------
# TODO:
# - [some-notes.txt]
#   Thu 13 Nov 14
#
#      wrote great linear sort
#
#   Sat 15 Nov 14
#
#      rewrote not-so-great n log n sort
#
# - [more-notes.txt]
#   Fri 14 Nov 14
#
#      ran unit tests
#
#-  [notes-missing-timestamp.txt]
#      read up on sorting
#
# Sample output:
#
#   Thurs 13 Nov 14
#
#      wrote great linear sort
#
#   Fri 14 Nov 14
#
#      ran unit tests
#
#   Sat 15 Nov 14
#
#       rewrote not-so-great n log n sort
#
#

"""Unit tests for merge_nootes module"""

import re
import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo


class TestIt(unittest.TestCase):
    """Class for testcase definition"""
    script_module = "merge_notes"        # name for invocation via 'python -m'

    def setUp(self):
        """Per-test initializations: disables tracing scripts invoked via run();
        initializes temp file name (With override from environment)."""
        # Note: By default, each test gets its own temp file.
        tpo.debug_print("setUp()", 4)
        gh.disable_subcommand_tracing()
        self.temp_file = tpo.getenv_text("TEMP_FILE",
                                         tempfile.NamedTemporaryFile().name)
        return

    def run_script(self, options, data_file):
        """Runs the script over the DATA_FILE, passing OPTIONS"""
        tpo.debug_format("run_script({opts}, {file})", 5,
                         opts=options, file=data_file)
        data_path = gh.resolve_path(data_file)
        output = gh.run("python  -m {module}  {opts}  {path}",
                        module=self.script_module, opts=options, path=data_path)
        # Make sure no python or bash errors. For example,
        #   "SyntaxError: invalid syntax" and "bash: python: command not found"
        error_found = re.search(r"(\S+Error:)|(no module)|(command not found)",
                                output.lower())
        self.assertFalse(error_found)
        traceback_found = re.search("Traceback.*most recent call", output)
        self.assertFalse(traceback_found)
        return output

    def test_data_file_TODO(self):
        """TODO: Tests results over a known data file"""
        tpo.debug_print("test_data_file()", 4)
        data_file = "TODO_filename"
        output = self.run_script("--TODO-options", data_file)
        self.assertTrue(re.search("TODO-regex", output))
        return

    def test_TODO_func(self):
        """TODO: Tests func(arg1-desc, ...)"""
        tpo.debug_print("test_TODO()", 4)
        self.assertTrue(2 + 2 == 5)
        return

    def test_TODO1(self):
        """TODO: Ensure whatever"""
        tpo.debug_print("test_TODO1()", 4)
        self.assertTrue(2 + 2 == 5)
        return

    def test_TODO2(self):
        """TODO: Ensure whatever too"""
        tpo.debug_print("test_TODO2()", 4)
        pass

    def test_TODO2(self):
        """TODO: Ensure whatever three"""
        tpo.debug_print("test_TODO3()", 4)
        self.fail("need to implement test")

    def tearDown(self):
        """Per-test cleanup: deletes temp file unless detailed debugging"""
        tpo.debug_print("tearDown()", 4)
        if not tpo.detailed_debugging():
            gh.run("rm -vf {file}*", file=self.temp_file)
        return

#------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
