#! /usr/bin/env python
#
# Test(s) for ../filter_random.py
#
# Notes:
# - Fill out TODO's below. Use numbered tests to order (e.g., test_1_usage).
# - This can be run as follows:
#   $ PYTHONPATH=".:$PYTHONPATH" python tests/test_simple_main_example.py
#

"""Unit tests for randoomize_lines module"""

import re
import unittest
from unittest_wrapper import TestWrapper

import glue_helpers as gh
import tpo_common as tpo


class TestIt(TestWrapper):
    """Class for testcase definition"""
    script_module = TestWrapper.derive_tested_module_name(__file__)
    count = 10

    def test_data_file(self):
        """Makes sure sample randomization works as expected """
        tpo.debug_print("TestIt.test_data_file()", 4)
        spanish_numbers = ["uno", "dos", "tres", "cuatro", "cinco",
                           "seis", "siete", "ocho", "nueve", "diez"]
        num_lines = 1 + len(spanish_numbers)
        HEADER = "word\tvalue"

        # Generate data
        data_lines = [HEADER]
        for i, word in enumerate(spanish_numbers):
            data_lines.append(word + "\t" + str(i + 1))
        gh.write_lines(self.temp_file, data_lines)

        # Do N trials fo see if all are permutations, with header included first
        num_good = 0
        options = "--include-header"
        for c in range(self.count):
            output = self.run_script(options, self.temp_file)
            self.assertTrue(output.startswith(HEADER))
            output_lines = output.split("\n")
            self.assertTrue(tpo.intersection(output_lines, data_lines))
            self.assertEquals(len(output_lines), len(data_lines))
        return

#------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
