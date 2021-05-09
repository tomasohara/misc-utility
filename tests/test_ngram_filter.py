#! /usr/bin/env python
#
# TODO: Test(s) for ../MODULE.py
#
# Notes:
# - This includes a test for the following:
#    $ head _dog_fleas.*
#    ==> _dog_fleas.bi <==
#    my dog	10
#    dog has	5
#    has fleas	1
#    
#    ==> _dog_fleas.txt <==
#    My dog has fleas very fat. Poor dog of mine!
#    
#    ==> _dog_fleas.uni <==
#    dog	1000
#    fleas	25
#    
#    $ python -m ngram_filter _dog_fleas.uni _dog_fleas.bi < _dog_fleas.txt
#    My dog has fleas very fat. Poor dog of mine! => ['my dog has fleas', 'dog']
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

    def test_data_file(self):
        """Makes sure ngram filter works as expected"""
        tpo.debug_print("TestIt.test_data_file()", 4)

        # Create the supporting data files along with input
        uni_data = ["dog\t1000", "fleas\t25", "dawg\t2"]
        uni_file = self.temp_file + ".uni"
        gh.write_lines(uni_file, uni_data)
        #
        bi_data = ["my dog\t10", "dog has\t25", "very fat\t5", "has fleas\t2"]
        bi_file = self.temp_file + ".bi"
        gh.write_lines(bi_file, bi_data)
        #
        text_data = ["My dog has fleas very fat.", "Poor dawg of mine!"]
        text_file = self.temp_file + ".txt"
        gh.write_lines(text_file, text_data)

        # Run the script and check for known ngrams with sufficient frequency
        options = uni_file + " " + bi_file
        # TODO: modify ngram_filter.py to accept optional filename for input
        output = self.run_script(options + " < " + text_file)
        expected_matches = ['my dog has fleas', 'very fat', 'dawg']
        match_list = re.findall("'([^']+)'", output)
        self.assertEquals(expected_matches, match_list)

#------------------------------------------------------------------------

if __name__ == '__main__':
    tpo.trace_current_context()
    unittest.main()
