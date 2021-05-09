#! /usr/bin/env python
#
# Test(s) for train_language_model.py
#
# Notes:
# - Automates the following steps:
#   $ {
#      PATH="$HOME/programs/kenlm/bin:$PATH"
#      copy tests/sample-query-log.retrieve_lr_inputs.data /tmp/sample-query-log.txt
#      train_language_model.py /tmp/sample-query-log.txt
#      grep -A3 '\-grams:$' /tmp/sample-query-log.3gram.arpa
#   }
# - The KenLM binaries directory should be in the path
# - This can be run as follows:
#   $ PATH="$HOME/programs/kenlm/bin:$PATH" PYTHONPATH="." python tests/test_train_language_model.py
#
# Sample output data being tested:
#
#   $ grep -A3 '\-grams:$' /tmp/sample-query-log.3gram.arpa
#   \1-grams:
#   -0.32700765	<unk>
#   -inf	<s>	-0.054403726
#   -4.0367923	</s>
#   --
#   \2-grams:
#   -3.5224903	0 </s>
#   -0.924277	<s> impression	-0.12958996
#   -2.967703	0 impression	-0.12958996
#   --
#   \3-grams:
#   -2.5101326	10.255.207.242' 0 </s>
#   -2.4358258	10.255.207.242' 0 impression
#   -0.20094964	<s> impression dist
#
#   $ wc -l /tmp/sample-query-log.3gram.arpa
#   3797 /tmp/sample-query-log.3gram.arpa
#   


import re
import sys
import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo

class TestIt(unittest.TestCase):
    """Class for testcase definition"""
    script_module_name = "train_language_model"  # name for invocation via 'python -m ...'

    def setUp(self):
        """Per-test initializations, including disabling tracing of scripts invoked via run()"""
        tpo.debug_print("setUp()", 4)
        gh.disable_subcommand_tracing()
        self.base = tpo.getenv_text("TEMP_FILE", tempfile.NamedTemporaryFile().name)
        return

    def run_script(self, options, data_file):
        """Runs the script over the DATA_FILE, passing OPTIONS"""
        tpo.debug_format("run_script({options}, {data_file})", 5)
        output = gh.run("python  -m {self.script_module_name}  {options}  {data_file}")
        self.assertTrue(not re.search("(error:)|(no module)", output.lower()))
        return (output)

    def test_data_file(self):
        """Tests results over a known data file"""
        tpo.debug_print("test_data_file()", 4)
        data_file = gh.resolve_path("sample-query-log.retrieve_lr_inputs.data")

        # Run language modelling over sample query
        base = self.base
        gh.run("cp -v {data_file} {base}.txt")
        output = self.run_script("--verbose --interpolate-unigrams", base + ".txt")
        language_model_file = base + ".3gram.arpa"
        self.assertTrue(re.search(language_model_file, output))
        language_model_lines = gh.read_lines(language_model_file)
        self.assertTrue(len(language_model_lines) > 3000)
        # ex: \1-grams:
        self.assertTrue(gh.extract_matches("([0-9])-grams:$", language_model_lines) == ['1', '2', '3'])
        #      probability	ngram		backoff weight
        # ex: -0.25090787	Windows NT	-1.1418762
        # ex: -0.2513883	'patent attorney'	-1.0066712
        windows_nt = tpo.safe_float(gh.extract_match("(\S+)\tWindows NT\t\S+", language_model_lines))
        patent_attorney = tpo.safe_float(gh.extract_match("(\S+)\t'patent attorney'\t\S+", language_model_lines))
        self.assertTrue(windows_nt > patent_attorney)
        return

    def tearDown(self):
        """Per-test cleanup: deletes temporary file unless during detailed debugging"""
        tpo.debug_print("tearDown()", 4)
        if not tpo.detailed_debugging():
            gh.run("rm -vf {self.base}*")
        return

if __name__ == '__main__':
    unittest.main()
