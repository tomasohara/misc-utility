#! /usr/bin/env python
#
# TODO: Test(s) for ../archive_dir.py
# - local dir target
# - s3 storage
# - remote host
#
# Notes:
# - Fill out TODO's below. Use numbered tests to order (e.g., test_1_usage).
# - This can be run as follows:
#   $ PYTHONPATH=".:$PYTHONPATH" python tests/test_archive_dir.py
#

"""Unit tests for archive_dir module"""

import re
import tempfile
import unittest

import glue_helpers as gh
import tpo_common as tpo


class TestIt(unittest.TestCase):
    """Class for testcase definition"""
    script_module = "archive_dir"        # name for invocation via 'python -m'

    def setUp(self):
        """Per-test initializations: disables tracing scripts invoked via run();
        initializes temp file name (With override from environment)."""
        # Note: By default, each test gets its own temp file.
        tpo.debug_print("setUp()", 4)
        gh.disable_subcommand_tracing()
        self.temp_file = tpo.getenv_text("TEMP_FILE",
                                         tempfile.NamedTemporaryFile().name)
        return

    def run_script(self, env_options, input_dir, target_dest):
        """Runs the script setting ENV_OPTIONS and passing ARGUMENTS"""
        tpo.debug_format("run_script({env}, {dir}, {dest))", 5,
                         env=env_options, dir=input_dir, dest=target_dest)
        output = gh.run("{env} python  -m {module}  {dir}  {dest}",
                        env=env_options, module=self.script_module, 
                        dir=input_dir, dest=target_dest)
        # Make sure no python or bash errors. For example,
        #   "SyntaxError: invalid syntax" and "bash: python: command not found"
        error_found = re.search(r"(\S+Error:)|(no module)|(command not found)",
                                output.lower())
        self.assertFalse(error_found)
        traceback_found = re.search("Traceback.*most recent call", output)
        self.assertFalse(traceback_found)
        return output

    def test_local_dir(self):
        """Tests results using a directory archived locally"""
        # TODO: rework in terms of new verbose output mode
        tpo.debug_print("test_local_dir()", 4)
        input_dir = "/etc"

        # Determine input usage
        # note: sample du outout: 
        #  du: cannot read directory `/etc/chatscripts': Permission denied
        #  10468	/etc
        etc_usage = gh.run("du {dir} 2>&1 | grep -v ^du: | cut -f1", 
                           dir=input_dir)
        etc_mb = tpo.safe_int(etc_usage) / 1024.0
        gh.assertion(etc_mb > 1)

        output = self.run_script("DEBUG_LEVEL=4 MAX_CHUNK_MB=1 ADD_SUBDIR_AFFIX=1 ARCHIVE_NAME='local-etc'", input_dir, "/tmp")
        # package_and_xfer(/etc/init.d, [{'/etc/X11': 500, '/etc/kernel/prerm.d': 8, ... '/etc/xdg/autostart': 128, '/etc/menu-methods': 24}])
        self.assertTrue(re.search("package_and_xfer\(/etc/init.d", output))
        tar_files = gh.get_matching_files("/tmp/*local-etc*tar.gz")
        self.assertTrue(len(tar_files) > 1)
        return

    def TODO_test_s3(self):
        """TODO: Ensure whatever"""
        tpo.debug_print("test_TODO()", 4)
        self.assertTrue(2 + 2 == 5)
        return

    def tearDown(self):
        """Per-test cleanup: deletes temp file unless detailed debugging"""
        tpo.debug_print("tearDown()", 4)
        if not tpo.detailed_debugging():
            gh.run("rm -vf {file}*", file=self.temp_file)
        return

#------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
