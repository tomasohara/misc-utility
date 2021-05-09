#! /usr/bin/env python
#
# Test for extract_html_table_data.py.
#
# Sample test performed:
#
#   $ egrep '<table>|<td>' tests/sample-data-table.html
#       <table>
#             <td>cat 1</td> 
#             <td>10</td> 
#             <td>cat 2</td> 
#             <td>20</td> 
#   
#   $ extract_html_table_data.py tests/sample-data-table.html
#   cat 1	10
#   cat 2	20
#
#------------------------------------------------------------------------
#
# Notes:
# - This can be run as follows:
#   $ PYTHONPATH="." python -m tests/extract_html_table_data
#

import re
import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo

class TestIt(unittest.TestCase):
    """Class for testcase definition"""
    script_module_name = "extract_html_table_data"

    def setUp(self):
        """Per-test initializations, including disabling tracing of scripts invoked via run()"""
        tpo.debug_print("setUp()", 4)
        gh.disable_subcommand_tracing()
        self.temp_file = tpo.getenv_text("TEMP_FILE", tempfile.NamedTemporaryFile().name)
        return

    def run_script(self, options, data_file):
        """Runs the script over the sample DATA_FILE, passing OPTIONS"""
        tpo.debug_format("run_master({options}, {data_file})", 5)
        data_file = gh.resolve_path(data_file)
        output = gh.run("python  -m {self.script_module_name}  {options}  {data_file}")
        self.assertTrue(not re.search("(error:)|(no module)", output.lower()))
        return (output)

    def test_extract_html_table_data(self):
        """Ensure a simple 2x2 HTML table gets extracted OK"""
        tpo.debug_print("test_extract_html_table_data()", 4)
        table = [
            "<html>",
            "  <body>",
            "    <table>",
            "      <tr>",
            "         <td>cat 1</td>",
            "         <td>10</td>",
            "      </tr>",
            "      <tr>",
            "        <td>cat 2</td>",
            "        <td>20</td>",
            "      </tr>",
            "    </table>",
            "  </body>",
            "</html>",
            ]
        gh.write_lines(self.temp_file, table)
        output = self.run_script("", self.temp_file)
        self.assertTrue(re.search("cat 1\t10\n", output))
        self.assertTrue(re.search("cat 2\t20\n", output))
        return
        
if __name__ == '__main__':
    unittest.main()
