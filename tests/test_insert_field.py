#! /usr/bin/env python
#
# Test(s) for ../insert_field.py
#
# Notes:
# - This can be run as follows:
#   $ PYTHONPATH=".:$PYTHONPATH" python tests/test_insert_field.py
#

"""Unit tests for insert_field module"""

import re
import sys
import unittest
from unittest_wrapper import TestWrapper

import glue_helpers as gh
import tpo_common as tpo


class TestIt(TestWrapper):
    """Class for testcase definition"""
    script_module = TestWrapper.derive_tested_module_name(__file__)

    def test_data_file(self):
        """Makes sure field insertion works as expected"""
        tpo.debug_print("TestIt.test_data_file()", 4)

        # Create external lookup table to use for augmentation
        # note: Order doesn't matter in lookup table (except for header)
        table_data = ["Num\tEnglish",
                      "3\tThree",
                      "2\tTwo",
                      "1\tOne"]
        table_file = self.temp_base + ".lookup.data"
        gh.write_lines(table_file, table_data)
        shelve_file = self.temp_base + ".shelve"
        # note: uses shelve so that can be loaded as part of constructor
        gh.run("USE_SHELVE=1 python -m table_lookup  --skip-test  --save-file {outf}  {inf}",
               outf=shelve_file, inf=table_file)

        # Add above fields to existing table
        test_data = ["Num\tSpanish\tFrench",
                     "1\tUno\tUn",
                     "2\tDos\tDeux",
                     "3\tTres\tTrois"]
        test_file = self.temp_base + ".test.data"
        gh.write_lines(test_file, test_data)
        output = self.run_script("--update-headers --key-field-num 1 --new-field-num 2 " + shelve_file,
                                 test_file, env_options="USE_SHELVE=1")
        output_lines = output.split("\n")
        tpo.debug_format("output_lines={ol}", 4, ol=output_lines)

        # Do bunch of saniry checks
        self.assertTrue(len(output_lines) == 4)
        self.assertTrue(len(output_lines[0].split("\t")) == 4)
        self.assertTrue(re.search("Num.English.Spanish.French", output_lines[0]))
        self.assertTrue(re.search("1.One.Uno.Un", output_lines[1]))
        self.assertTrue(re.search("2.Two.Dos.Deux", output_lines[2]))
        self.assertTrue(re.search("3.Three.Tres.Trois", output_lines[3]))
        return

#------------------------------------------------------------------------

if __name__ == '__main__':
    tpo.trace_current_context()
    unittest.main()
    if tpo.debugging() and "--help" in sys.argv:
        print("Environment options:")
        print("\t" + tpo.formatted_environment_option_descriptions())
        
