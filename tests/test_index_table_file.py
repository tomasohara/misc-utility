#! /usr/bin/env python
#
# Test(s) for index_table_file.py. The main test indexes the sample consolidated data file
# and makes sure index files created and that simple search works as expected.
#
# ------------------------------------------------------------------------
# Sample input:
#
#   impression	dist	context	norm	score	date	titlelen_chars	descriptionlen	age	sort_type	sponsored	k	l	c	rpos	title	site	delta_age	cfi_score	cfi_type	probability	algorithm	site_type	r	query_norm	include_outside_geospec	job_count	age_count	delta_dists	delta_ages	page_uuid	session	format	page_num	posting_id	sponsored_job_count	user_agent	user_geoloc_id	gross_bid	cost	ip	clicked
#   1aa61096dd9a4126b4ecf6efcfd05605	0	search	0	0	2013-01-17 23:42:48 UTC	0	0	-1	search,relevance,date	0	'biostatistics'	''	''	-1	0	0	0	0	0	0	1:'12396A'	0	''	None	False	1312	0	0	0	rpOdpj2wQr6r6AfmeEPd_A	8a9b7811e9c1982e69074e902880aec9e9c6a0f7	rss	1	0	0	'Mozilla/4.0 (compatible; Windows;)'		0	0	'101.226.33.180, 10.255.207.242'	0
#
#------------------------------------------------------------------------
# Notes:
# - This can be run as follows:
#   $ PYTHONPATH="." python tests/test_index_table_file.py
#

import re
import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo


class TestIt(unittest.TestCase):
    """Class for testcase definition"""
    script_module_name = "index_table_file"       # name for invocation via 'python -m ...'

    def setUp(self):
        """Per-test initializations: disables tracing of scripts invoked via run(); initializes temp file name (With override from environment)."""
        # Note: By default, each test gets its own temp file.
        tpo.debug_print("setUp()", 4)
        gh.disable_subcommand_tracing()
        self.temp_file = tpo.getenv_text("TEMP_FILE", tempfile.NamedTemporaryFile().name)
        # Get rid of any files left from previous run (e.g., with detailed debugging)
        gh.run("rm -rvf {self.temp_file}*")
        return

    def run_script(self, options, index_dir, data_file):
        """Runs the script over the DATA_FILE, passing OPTIONS"""
        # note: Usage: python ./index_table_file.py table_file [index_dir] [append]
        assert(options == "")
        tpo.debug_format("run_script({options}, {data_file})", 5)
        output = gh.run("python  -m {self.script_module_name}  {options}  {data_file}  {index_dir}")
        self.assertTrue(not re.search("(error:)|(no module)", output.lower()))
        return (output)

    def test_data_file(self):
        """Tests results over a known data file"""
        tpo.debug_print("test_data_file()", 4)
        data_file = gh.resolve_path("sample-query-log.retrieve_lr_inputs.data")
        index_dir = self.temp_file + "-index"
        # Index the sample data file
        output = self.run_script("", index_dir, data_file)
        self.assertTrue(re.search("commiting.*done", output))
        # Make sure index directory has expected files
        output = gh.run("ls {index_dir}")
        self.assertTrue(re.search("_0\.cfs", output)) 
        self.assertTrue(re.search("segments\.gen", output)) 
        # Test that first column used as mini-document ID field
        output = gh.run("echo 'biostatistics' | python -m search_table_file_index {index_dir}")
        self.assertTrue(output == "1aa61096dd9a4126b4ecf6efcfd05605")
        return
        
    def tearDown(self):
        """Per-test cleanup: deletes temporary file unless during detailed debugging"""
        tpo.debug_print("tearDown()", 4)
        if (tpo.debugging_level() < 4):
            gh.run("rm -rvf {self.temp_file}*")
        return


#------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
