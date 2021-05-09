#! /usr/bin/env python
#
# Test(s) for ../split_file.py
#
# Notes:
# - This can be run as follows:
#   $ PYTHONPATH=".:$PYTHONPATH" python tests/test_split_file.py
#

"""Unit tests for split_file module"""

import re
import unittest
from unittest_wrapper import TestWrapper

import glue_helpers as gh
import tpo_common as tpo


class TestIt(TestWrapper):
    """Class for testcase definition"""
    script_module = TestWrapper.derive_tested_module_name(__file__)
    NUM_LINES = 100
    NUM_SPLITS = 3

    def test_data_file(self):
        """Makes sure input file split propoerly"""
        tpo.debug_print("TestIt.test_data_file()", 4)

        # Setup data and run script
        data = [str(i) for i in range(self.NUM_LINES)]
        gh.write_lines(self.temp_file, data)
        output_base = self.temp_base + "-split"
        options = tpo.format("--num-splits {s} --output-base {b}", 
                             s=self.NUM_SPLITS, b=output_base)
        output = self.run_script(options, self.temp_file)
        self.assertTrue(re.search(r"split-0.*split-2",
                                  output.strip(), re.DOTALL))
        # Read split files 
        split_filenames = output.split()
        split_lines = []
        for i in range(self.NUM_SPLITS):
            split_lines.append(gh.read_lines(split_filenames[i]))

        # Check for expected content
        self.assertTrue(split_lines[0][0] == '0')
        self.assertTrue(len(split_lines[0]) == 34)
        self.assertTrue(len(split_lines[1]) == 33)
        self.assertTrue(split_lines[1][0] == '1')
        self.assertTrue(split_lines[2][0] == '2')
        self.assertTrue(len(split_filenames) == 3)
        return

#------------------------------------------------------------------------

if __name__ == '__main__':
    tpo.trace_current_context()
    unittest.main()
