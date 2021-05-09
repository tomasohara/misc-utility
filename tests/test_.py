#! /usr/bin/env python
#
# Tests that are not module specific (e.g., flag dash usage in filenames).
#
# Notes:
# - This is normally run from the parent directory, such as follows:
#   $ PYTHONPATH=".:$PYTHONPATH" python tests/test_.py
#

"""Tests that are not module specific"""

import glob
import os
import re
import unittest
from unittest_wrapper import TestWrapper

import debug
import system


class TestIt(TestWrapper):
    """Class for testcase definition"""
    script_module = None

    def test_dash_usage(self):
        """Report error if any filename has a dash (instead of underscore)"""
        scripts_with_dashes = glob.glob("*-*.py")
        self.assertEqual(scripts_with_dashes, [])


#------------------------------------------------------------------------

if __name__ == '__main__':
    debug.trace_current_context()
    if (os.path.basename(os.getcwd()) == "tests"):
        system.print_stderr("Warning: this should be run from parent directory")
    unittest.main()
