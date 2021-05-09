#! /usr/bin/env python
#
# Test(s) for ../simple_main_example.py
#
# Notes:
# - This can be run as follows:
#   $ PYTHONPATH=".:$PYTHONPATH" python tests/test_simple_main_example.py
#

"""Unit tests for simple_main_example module"""

import re
import unittest
from unittest_wrapper import TestWrapper

import glue_helpers as gh
import tpo_common as tpo


class TestIt(TestWrapper):
    """Class for testcase definition"""
    script_module = TestWrapper.derive_tested_module_name(__file__)
    unicode_sample_file = "tomas.txt"

    def setUp(self):
        """Post-argument parsing processing: just displays context"""
        # Note: This is done for contrast with test_main.py.
        tpo.debug_format("setup(): self={s}", 5, s=self)
        super(TestIt, self).setUp()
        tpo.trace_current_context(level=tpo.QUITE_DETAILED)

    def test_simple_data(self):
        """Make sure simple data sample processed OK"""
        gh.write_lines(self.temp, "really fubar")
        output = self.run_script("--fubar")
        self.assertTrue("really" in output)

    def test_data_file(self):
        """Makes sure sample filter works as expected"""
        tpo.debug_print("test_data_file()", 4)
        data = ["a", "fubar", "b", "fubar", "c"]
        gh.write_lines(self.temp_file, data)
        output = self.run_script("", self.temp_file)
        self.assertTrue(re.search(r"^fubar\s+fubar$", 
                                  output.strip()))
        return

    def test_unicode(self):
        """Make sure unicode is output OK"""
        # TODO: create scriptlet with Main.force_unicode enabled
        tpo.debug_print("test_unicode()", 4)
        data_path = gh.resolve_path(self.unicode_sample_file)
        gh.copy_file(data_path, self.temp_file)
        output = self.run_script("--regex hola", self.temp_file)
        # Check for inverted exclamation, "hola",  and regular exclamation
        # C2A1    U+00a1  Po      INVERTED EXCLAMATION MARK
        self.assertTrue(re.search("\xC2\xA1.*!", 
                                  output.strip()))

#------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
