#! /usr/bin/env python
#
# Test(s) for transpose_data.py. This can be run as follows:
# $ PYTHONPATH="." tests/test_transpose_data.py
#

import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo

class TestIt(unittest.TestCase):
    """Class for testcase definition"""
    input_data = [
        "COL1\tCOL2", 
        "R1C1\tR1C2", 
        "R2C1\tR2C2"]

    def setUp(self):
        """Per-test initializations, including disabling tracing of scripts invoked via run()"""
        tpo.debug_print("setUp()", 4)
        gh.disable_subcommand_tracing()
        return

    def run(self, data, options=None):
        """Run script with OPTIONS over DATA lines, returning output list"""
        # Setup input
        # TODO: extend run to accept file input
        temp_file = tempfile.NamedTemporaryFile().name
        gh.write_lines(temp_file, self.input_data)
        # Get the transposed data and verify
        output = gh.run("python -m transpose_data {opt} < {inp}",
                        opt=(options or ""), inp=temp_file)
        return output.split("\n")

    def test1_simple(self):
        """Tests simple transpose output"""
        tpo.debug_print("TestIt.test1_simple()", 5)
        expected_output = [
            "COL1\tR1C1\tR2C1", 
            "COL2\tR1C2\tR2C2",
            ]
        output = self.run(self.input_data)
        self.assertEqual(expected_output, output)

    def test2_single(self):
        """Tests single-field output"""
        tpo.debug_print("TestIt.test2_single()", 5)
        expected_output = [
            "COL1\tR1C1",
            "COL2\tR1C2",
            "COL1\tR2C1",
            "COL2\tR2C2",
            ]
        output = self.run(self.input_data, options="--single-field")
        self.assertEqual(expected_output, output)

if __name__ == '__main__':
    unittest.main()
