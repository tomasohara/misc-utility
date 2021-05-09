#! /usr/bin/env python
#
# Test(s) for spell.py. This can be run as follows:
# $ PYTHONPATH="." python -u tests/test_spell.py
#

import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo

class TestIt(unittest.TestCase):
    """Class for testcase definition"""

    def setUp(self):
        """Per-test initializations, including disabling tracing of scripts invoked via run()"""
        tpo.debug_print("setUp()", 4)
        gh.disable_subcommand_tracing()
        return

    def test_known_misspellings(self):
        """Check that sample bad spellings flagged as well as that sample good ones not flagged"""
        bad_spellings = ["seperate", "dawg", "recieve"]
        good_spellings = ["sincere", "economical", "fortnight"]
        # Setup input
        temp_file = tempfile.NamedTemporaryFile().name
        gh.write_lines(temp_file, bad_spellings + good_spellings)
        # Get the misspelled words and verify
        output = gh.run("python -m spell {temp_file}")
        self.assertEqual(output.split("\n"), bad_spellings)
        return

if __name__ == '__main__':
    unittest.main()
