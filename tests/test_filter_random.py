#! /usr/bin/env python
#
# Test(s) for ../filter_random.py
#
# Notes:
# - Fill out TODO's below. Use numbered tests to order (e.g., test_1_usage).
# - This can be run as follows:
#   $ PYTHONPATH=".:$PYTHONPATH" python tests/test_simple_main_example.py
#

"""Unit tests for filter_random module"""

import re
import unittest
from unittest_wrapper import TestWrapper

import glue_helpers as gh
import tpo_common as tpo


def is_sublist(list1, list2):
    """Verifies that the items in LIST1 and all contained in LIST2 in the same order"""
    # EX: is_sublist([1, 3, 5], [1, 2, 3, 4, 5])
    # EX: not is_sublist([5, 3, 1], [1, 2, 3, 4, 5])
    ok = True
    last_pos = -1
    for pos1, item1 in enumerate(list1):
        if item1 not in list2:
            ok = False
            break
        else:
            pos2 = list2.index(item1)
            if pos2 < last_pos:
                ok = False
                break
            last_pos = pos2
            gh.assertion(pos1 <= pos2)
    tpo.debug_format("is_sublist({l1}, {l2}) = {r}", 5, 
                     r=ok, l1=list1, l2=list2)
    return ok


class TestIt(TestWrapper):
    """Class for testcase definition"""
    script_module = TestWrapper.derive_tested_module_name(__file__)
    COUNT = 10

    def test_data_file(self):
        """Makes sure sample filter works as expected """
        tpo.debug_print("TestIt.test_data_file()", 4)
        spanish_numbers = ["uno", "dos", "tres", "cuatro", "cinco",
                           "seis", "siete", "ocho", "nueve", "diez"]
        num_lines = 1 + len(spanish_numbers)
        HEADER = "word\tvalue"
        #
        RATIO = 0.25
        min_lines = min(2, 1 + (RATIO / 2))
        max_lines = max(num_lines - 1, 1 + (RATIO / 2))

        # Generate data
        data_lines = [HEADER]
        for i, word in enumerate(spanish_numbers):
            data_lines.append(word + "\t" + str(i + 1))
        gh.write_lines(self.temp_file, data_lines)

        # Do N trials fo see if most are proper subsets, with header included
        num_good = 0
        options = "--include-header --ratio {r}".format(r=RATIO)
        for c in range(self.COUNT):
            output = self.run_script(options, self.temp_file)
            self.assertTrue(output.startswith(HEADER))
            output_lines = output.split("\n")
            self.assertTrue(is_sublist(output_lines, data_lines))
            ok = (min_lines <= len(output_lines) <= max_lines)
            if ok:
                num_good += 1
            tpo.debug_format("trial {num}: {ok}", 4, num=(c + 1), ok=ok)
        tpo.debug_format("{n} of {c} good: {pct}%", 4,
                         n=num_good, c=self.COUNT, 
                         pct=tpo.round_num(num_good / self.COUNT))
        self.assertTrue(num_good >= (self.COUNT / 2))
        return

#------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
