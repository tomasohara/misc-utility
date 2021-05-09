#! /usr/bin/env python
#
# TODO: Test(s) for ../MODULE.py
#
# Notes:
# - Fill out TODO's below. Use numbered tests to order (e.g., test_1_usage).
# - TODO: If any of the setup/cleanup methods defined, make sure to invoke base
#   (see examples below for setUp and tearDown).
# - This can be run as follows:
#   $ PYTHONPATH=".:$PYTHONPATH" python tests/test_MODULE.py
#

"""TODO: Unit tests for MODULE module"""

import re
import unittest
from unittest_wrapper import TestWrapper

import glue_helpers as gh
import tpo_common as tpo


class TestIt(TestWrapper):
    """Class for testcase definition"""
    script_module = TestWrapper.derive_tested_module_name(__file__)
    # TODO: use_temp_base_dir = True            # treat TEMP_BASE as directory

    ## TODO: optional setup methods
    ##
    ## @classmethod
    ## def setUpClass(cls):
    ##     tpo.debug_print("TestIt.setUpClass()", 6)
    ##     super(TestIt, cls).setUpClass()
    ##     ...
    ##
    ## def setUp(self):
    ##     tpo.debug_print("TestIt.setUp()", 6)
    ##     super(TestIt, self).setUp()
    ##     ...
    ##

    def test_data_file(self):
        """Makes sure TODO works as expected"""
        tpo.debug_print("TestIt.test_data_file()", 4)
        data = ["TODO1", "TODO2"]
        gh.write_lines(self.temp_file, data)
        output = self.run_script("", self.temp_file)
        self.assertTrue(re.search(r"TODO-pattern", 
                                  output.strip()))
        return

    def test_something_else(self):
        """TODO: flesh out test for something else"""
        self.fail("TODO: code test")

    ## TODO: optional cleanup methods
    ##
    ## def tearDown(self):
    ##     tpo.debug_print("TestIt.tearDown()", 6)
    ##     super(TestIt, cls).tearDownClass()
    ##     ...
    ##
    ## @classmethod
    ## def tearDownClass(cls):
    ##     tpo.debug_print("TestIt.tearDownClass()", 6)
    ##     super(TestIt, self).tearDown()
    ##     ...
    ##

#------------------------------------------------------------------------

if __name__ == '__main__':
    tpo.trace_current_context()
    unittest.main()
