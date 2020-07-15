#! /usr/bin/env python
#
# Wrapper class around unittest.TestCase
#
# Notes:
# - Based on template.py used in older test wcripts
# - Creates per-test temp file, based on same class-wide temp-base file.
# - To treat temp-based as a subdirectory. set use_temp_base_dir to True in 
#   class member initialiation section.
# - Changes to temporary directory/file should be synchronized with ../main.py.
# - Overriding the temporary directory can be handy during debugging; however,
#   you might need to specify different ones if you invoke helper scripts. See
#   tests/test_extract_company_info.py for an example.
#
#-------------------------------------------------------------------------------
# Sample test (streamlined version of test_simple_main_example.py):
#
#   import unittest 
#   from unittest_wrapper import TestWrapper
#   import glue_helpers as gh
#
#   class TestIt(TestWrapper):
#       """Class for testcase definition"""
#       script_module = TestWrapper.derive_tested_module_name(__file__)
#  
#       def test_simple_data(self):
#           """Make sure simple data sample processed OK"""
#           gh.write_lines(self.temp, "really fubar")
#           output = self.run_script("--fubar")
#           self.assertTrue("really" in output)
#
#   if __name__ == '__main__':
#      unittest.main()
#-------------------------------------------------------------------------------
# TODO:
# - add method to invoke unittest.main(), so clients don't need to import unittest
#

"""Unit test support class"""

import os
import re
import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo


class TestWrapper(unittest.TestCase):
    """Class for testcase definition"""
    script_module = "TODO MODULE"        # name for invocation via 'python -m'
    temp_base = tpo.getenv_text("TEMP_BASE",
                                tempfile.NamedTemporaryFile().name)
    use_temp_base_dir = None
    test_num = 1

    @classmethod
    def setUpClass(cls):
        """Per-class initialization: make sure script_module set properly"""
        tpo.debug_format("TestWrapper.setupClass(): cls={c}", 5, c=cls)
        super(TestWrapper, cls).setUpClass()
        tpo.trace_object(cls, 5, "TestWrapper class")
        if cls.script_module:
            help_usage = gh.run("python -m '{mod}' --help", mod=cls.script_module)
            gh.assertion("No module named" not in help_usage)

        # Optionally, setup temp-base directory (normally just a file)
        if cls.use_temp_base_dir is None:
            cls.use_temp_base_dir = tpo.getenv_bool("USE_TEMP_BASE_DIR", False)
            # TODO: temp_base_dir = tpo.getenv_text("TEMP_BASE_DIR", ""); cls.use_temp_base_dir = bool(temp_base_dir.strip); ...
        if cls.use_temp_base_dir:
            gh.run("mkdir -p {dir}", dir=cls.temp_base)

        return

    @staticmethod
    def derive_tested_module_name(test_filename):
        """Derive the name of the module being tested from TEST_FILENAME"""
        module = os.path.split(test_filename)[-1]
        module = re.sub(r".py[oc]?$", "", module)
        module = re.sub(r"^test_", "", module)
        tpo.debug_format("derive_tested_module_name({f}) = {m}", 5,
                         f=test_filename, m=module)
        return (module)

    def setUp(self):
        """Per-test initializations: disables tracing scripts invoked via run();
        initializes temp file name (With override from environment)."""
        # Note: By default, each test gets its own temp file.
        tpo.debug_print("TestWrapper.setUp()", 4)
        gh.disable_subcommand_tracing()
        # The temp file is an extension of temp-base file by default.
        # Opitonally, if can be a file in temp-base subdrectory.
        if self.use_temp_base_dir:
            default_temp_file = gh.form_path(self.temp_base, "test-")
        else:
            default_temp_file = self.temp_base + "-test-"
        default_temp_file += str(TestWrapper.test_num)
        self.temp_file = tpo.getenv_text("TEMP_FILE", default_temp_file)
        TestWrapper.test_num += 1
        tpo.trace_object(self, 5, "TestWrapper instance")
        return

    def run_script(self, options=None, data_file=None, log_file=None, 
                   out_file=None, env_options=""):
        """Runs the script over the DATA_FILE (optional), passing (positional)
        OPTIONS and optional setting ENV_OPTIONS. If OUT_FILE and LOG_FILE are
        not specifed, they  are derived from self.temp_file.
        Note: issues warning if script invocation leads to error"""
        tpo.debug_format("TestWrapper.run_script({opts}, {df}, {lf}, {of}, {env})", 5,
                         opts=options, df=data_file, lf=log_file, 
                         of=out_file, env=env_options)
        if options is None:
            options = ""

        # Derive the full paths for data file and log, and then invoke script.
        # TODO: derive from temp base and data file name?
        data_path = ""
        if data_file:
            data_path = gh.resolve_path(data_file)
        if not log_file:
            log_file = self.temp_file + ".log"
        if not out_file:
            out_file = self.temp_file + ".out"
        # note: output is redirected to a file to preserve tabs
        gh.issue("{env} python  -m {module}  {opts}  {path} 1> {out} 2> {log}",
                 env=env_options, module=self.script_module, 
                 opts=options, path=data_path, out=out_file, log=log_file)
        output = gh.read_file(out_file)
        # note; trailing newline removed as with shell output
        if output.endswith("\n"):
            output = output[:-1]
        tpo.debug_format("output: {{\n{out}\n}}", 4,
                         out=gh.indent_lines(output))

        # Make sure no python or bash errors. For example,
        #   "SyntaxError: invalid syntax" and "bash: python: command not found"
        log_contents = gh.read_file(log_file)
        error_found = re.search(r"(\S+error:)|(no module)|(command not found)",
                                log_contents.lower())
        gh.assertion(not error_found)
        tpo.debug_format("log contents: {{\n{log}\n}}", 5,
                         log=gh.indent_lines(log_contents))

        # Do sanity check for python exceptions
        traceback_found = re.search("Traceback.*most recent call", log_contents)
        gh.assertion(not traceback_found)

        return output

    def tearDown(self):
        """Per-test cleanup: deletes temp file unless detailed debugging"""
        tpo.debug_print("TestWrapper.tearDown()", 4)
        if not tpo.detailed_debugging():
            gh.run("rm -vf {file}*", file=self.temp_file)
        return

    @classmethod
    def tearDownClass(cls):
        """Per-class cleanup: stub for tracing purposes"""
        tpo.debug_format("TestWrapper.tearDownClass(); cls={c}", 5, c=cls)
        if not tpo.detailed_debugging():
            if cls.use_temp_base_dir:
                gh.run("rm -rvf {dir}", dir=cls.temp_base)
            else:
                gh.run("rm -vf {base}*", base=cls.temp_base)
        super(TestWrapper, cls).tearDownClass()
        return
