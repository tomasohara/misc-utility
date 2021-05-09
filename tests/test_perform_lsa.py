#! /usr/bin/env python
#
# Tests for perform_lsa.py. This includes a simple test based on Deerwester, et al.
# (1990), the classic paper on LSA, as well as a real-world test over job positngs.
#
# Notes:
# - S. Deerwester , S. Dumais , G. Furnas , T. Landauer, and R. Harshman (1990),
#   "Indexing by latent semantic analysis",  in Journal of the American Society
#   for Information Science, 
# - This can be run as follows:
#   $ PYTHONPATH="." python tests/test_perform_lsa.py
# - When checking the similarity results, use the version of the sample data which includes
#   a dummy posting ID (random100-titles-descriptions.list).
#------------------------------------------------------------------------
# Sample script invocation for tests
#
# $ python -m gensim_test --tfidf --save random100-titles-descriptions.txt
#
# $ python -m gensim_test --tfidf --load --similar-docs-of '72' random100-titles-descriptions
#
# $ NUM_TOPICS=10 python -m perform_lsa  --similar-docs-of '72' random100-titles-descriptions
#------------------------------------------------------------------------
# TODO:
# - Index distinct-job-sample (e.g., with num topics 5).
# - Verify that topics overlap with categories for test_data_file.
#

import re
import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo


TEST_ALL = tpo.getenv_boolean("TEST_ALL", not tpo.debugging(), "Runs all unit tests, including ones with platform differences")

class TestIt(unittest.TestCase):
    """Class for testcase definition"""
    script_module_name = "perform_lsa"  # name for invocation via 'python -m ...'

    def setUp(self):
        """Per-test initializations: disables tracing of scripts invoked via run(); initializes temp file name (With override from environment)."""
        # Note: By default, each test gets its own temp file.
        tpo.debug_print("setUp()", 4)
        gh.disable_subcommand_tracing()
        self.temp_base = tpo.getenv_text("TEMP_FILE", tempfile.NamedTemporaryFile().name)
        return

    def run_script(self, options, data_file):
        """Runs the script over the DATA_FILE, passing OPTIONS"""
        # Note: the script (perform_lsa.py) is run with random seed specified for reproducibility
        tpo.debug_format("run_script({options}, {data_file})", 5)
        output = gh.run("RANDOM_SEED=13  python  -m {self.script_module_name}  {options}  {data_file}")
        # Make sure no python or bash errors
        # examples: "SyntaxError: invalid syntax", "bash: python: command not found"
        self.assertTrue(not re.search("(\S+Error:)|(no module)|(command not found)", output.lower()))
        return (output)

    def run_LSA_test(self, filename, source_doc, similar_doc, num_topics=100, max_similar=10):
        """Helper function for tests that runs regular vector indexing and LSA-based indexing over FILENAME verifying that SOURCE_DOC matches SIMILAR_DOC only when latent semantic indexing used"""
        # Produce similarity via regular vector matching
        # ex: Documents similar to 71: [(71, '1.000'), (11, '0.847'), (3, '0.847'), (59, '0.826'), (42, '0.826'), (33, '0.826'), (19, '0.821'), (22, '0.821'), (45, '0.819'), (81, '0.816')]
        # note: Quirk in gensim_test.py requiring the model to be saved prior to similarity
        gh.assertion(filename.endswith(".txt"))
        gh.run("python -m gensim_test --save {filename}")
        data_file_base = gh.remove_extension(filename, ".txt")
        regular_output = gh.run("python  -m gensim_test  --load  --similar-docs-of {source_doc}  --max-similar {max_similar}  {data_file_base}")
        regular_sims = gh.extract_match_from_text(tpo.format("Documents similar to {source_doc}: \[([^\n]+)\]"), regular_output)
        tpo.debug_format("regular_sims={rs}", 4, rs=regular_sims)
        self.assertTrue(not re.search(tpo.format("\({similar_doc}, "), regular_sims))
        # Produce similarity via LSA
        # ex: Documents similar to 71: [(0, '0.613'), (35, '0.504'), (10, '0.502'), (39, '0.502'), (41, '0.502'), (73, '0.502'), (59, '0.501'), (75, '0.501'), (95, '0.501'), (7, '0.501')]
        lsa_output = self.run_script(tpo.format("--preprocess  --num-topics {num_topics}  --max-similar {max_similar}  --similar-docs-of {source_doc}"), data_file_base)
        lsa_sims = gh.extract_match_from_text(tpo.format("Documents similar to {source_doc}: \[([^\n]+)\]"), lsa_output)
        tpo.debug_format("lsa_sims={ls}", 4, ls=lsa_sims)
        self.assertTrue(re.search(tpo.format("\({similar_doc}, "), lsa_sims))
        return

    def test_deerwester_data(self):
        """Checks for document vector similarity over data from Deerwester et al. (1990)"""
        tpo.debug_print("test_deerwester_data()", 4)
        deerwester_data = [
            # Texts related to human-computer interaction (#'s 0-4)
            "Human machine interface for Lab ABC computer applications",
            "A survey of user opinion of computer system response time",
            "The EPS user interface management system",
            "System and human system engineering testing of EPS"
            "Relation of user-perceived response time to error measurement",
            # Texts related to graph theory (#'s 5-8)
            "The generation of random, binary, unordered trees",
            "The intersection graph of paths in trees",
            "Graph minors IV: Widths of trees and well-quasi-ordering",
            "Graph minors: A survey"
            ]
        temp_data_file = self.temp_base + ".deerwester.txt"
        ## BAD: gh.write_lines(temp_data_file, deerwester_data)
        ## HACK: doubles the data to avoid pruning during similarity calculations
        ## group 1: 0-4 & 9-13; group 2: 5-8 & 14-17
        gh.write_lines(temp_data_file, deerwester_data + deerwester_data)
        ## TODO: self.run_LSA_test(temp_data_file, 5, 8, num_topics=2, max_similar=3)
        self.run_LSA_test(temp_data_file, 5, 8, num_topics=10, max_similar=10)
        return

    def test_data_file(self):
        """Checks for document vector similarity that requires LSA to reveal"""
        # TODO: make the test more robust with respect to platform differences
        tpo.debug_print("test_data_file()", 4)
        if not TEST_ALL:
            tpo.print_stderr("Warning: Ignoring test due to potential platform differences\n")
            return
        data_file = gh.resolve_path("random100-titles-descriptions.txt")
        temp_data_file = self.temp_base + ".txt"
        gh.copy_file(data_file, temp_data_file)
        self.run_LSA_test(temp_data_file, 71, 95, num_topics=10, max_similar=10)
        return

    def tearDown(self):
        """Per-test cleanup: deletes temporary file unless during detailed debugging"""
        tpo.debug_print("tearDown()", 4)
        if (tpo.debugging_level() < 4):
            gh.run("rm -vf {self.temp_base}*")
        return

#------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
