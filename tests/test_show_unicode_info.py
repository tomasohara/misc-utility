#! /usr/bin/env python
#
# Test(s) for show_unicode_info.py
#
# Notes:
# - Fill out the TODO's through the file.
# - This can be run as follows:
#   $ PYTHONPATH="." python tests/test_show_unicode_info.py
#

import re
import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo


class TestIt(unittest.TestCase):
    """Class for testcase definition"""
    script_module_name = "show_unicode_info"  # name for invocation via 'python -m ...'

    def run_script(self, options, data_file):
        """Runs the script over the DATA_FILE, passing OPTIONS"""
        tpo.debug_format("run_script({options}, {data_file})", 5)
        output = gh.run("python  -m {self.script_module_name}  {options}  - < {data_file}")
        self.assertTrue(not re.search("(error:)|(no module)", output.lower()))
        return (output)

    def test_data_file(self):
        """Tests results over a known data file"""
        tpo.debug_print("test_data_file()", 4)
        data_file = gh.resolve_path("AR-arabic.txt")
        output = self.run_script("", data_file)
        self.assertTrue(re.search("EFBBBF.*ZERO WIDTH NO-BREAK SPACE.*U\+0627.*ARABIC LETTER ALEF.*U\+0629.*ARABIC LETTER TEH MARBUTA", output, re.MULTILINE|re.DOTALL))
        self.assertTrue(len(output.split("\n")) > 17)
        return
        
#------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

