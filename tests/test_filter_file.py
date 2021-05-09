#! /usr/bin/env python
#
# Test(s) for filter_file.py. This can be run as follows:
# $ PYTHONPATH="." python tests/test_filter_file.py
#

import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo

class TestIt(unittest.TestCase):
    """Class for testcase definition"""

    def setUp(self):
        """Per-test initializations: create dummy files and disables tracing of scripts invoked via run"""
        tpo.debug_print("setUp()", 4)
        self.init_tests()
        gh.disable_subcommand_tracing()
        return

    def init_tests(self):
        """Common one-time initialization for tests"""
        tpo.debug_print("init_tests()", 4)
        self.all_cases = [str(i) for i in range(10)]
        self.odd_cases = [str(i) for i in self.all_cases if (int(i) % 2) == 1]
        self.even_cases = [str(i) for i in self.all_cases if (int(i) % 2) == 0]

        # Setup up files with input
        self.data_file = tempfile.NamedTemporaryFile().name
        gh.write_lines(self.data_file, self.all_cases)
        self.filter_file = tempfile.NamedTemporaryFile().name
        gh.write_lines(self.filter_file, self.odd_cases)
        self.temp_file = tempfile.NamedTemporaryFile().name
        return

    def test_basic_filters(self):
        """Verify basic inclusion and exclusion filter work"""
        tpo.debug_print("test_basic_filters()", 4)
        output = gh.run("python -m filter_file --include {self.filter_file} < {self.data_file}")
        self.assertEqual(output.split("\n"), self.odd_cases)
        output = gh.run("python -m filter_file --exclude {self.filter_file} < {self.data_file}")
        self.assertEqual(output.split("\n"), self.even_cases)
        return

    def test_regex(self):
        """Verify simple regex patterns work"""
        tpo.debug_print("test_regex()", 4)
        output = gh.run("python -m filter_file --regex '^(\\d+)$' --include {self.data_file} < {self.data_file}")
        self.assertEqual(output.split("\n"), self.all_cases)
        output = gh.run("python -m filter_file --regex '^(\\d*[02468])$' --include {self.data_file} < {self.data_file}")
        self.assertEqual(output.split("\n"), self.even_cases)
        return

    def test_filter_field(self):
        """Verify that filtering by field number works as expected"""
        data_file = gh.resolve_path("query-log.session.filtered.data")
        # note: Filters input to just include two common queries (key field is 12th in consolidated query output),
        # which should produce 23 cases (out of 92).
        gh.write_lines(self.temp_file, 
                       ["'RADIATION ONCOLOGY'",
                        "'thermal phd'",
                        ])
        regex_output = gh.run("python  -m filter_file  --regex='^(?:[^\\t]*\\t){{11}}([^\\t]+)'  --include  {self.temp_file} < {data_file}")
        field_output = gh.run("python  -m filter_file  --field-num=12  --include  {self.temp_file} < {data_file}")
        self.assertTrue(len(field_output.split("\n")) > 20)
        self.assertEqual(regex_output, field_output)
        return

#------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
