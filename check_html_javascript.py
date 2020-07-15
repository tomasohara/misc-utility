#! /usr/bin/env python
#
# Run embedded JavaScript from HTML page through code checking tools, such as
# jslint and jshint. This includes common definitions such as for DOM object
# (e.g., window) and JQuery ($ selector function). It also optionally runs
# in strict mode to help check other potential errors (e.g., undefined vars).
#
#

"""Run JavaScript embedded in <script> tags through lint-style code checkers"""

from main import Main
import debug
import glue_helpers as gh
from regex import my_re
import system
import tempfile


CODE_CHECKERS = "code-checkers"
STRIP_INDENT = "strip-indent"
SKIP_SAFE_MODE = "skip-safe-mode"
SKIP_COMMON_DEFINES = "skip-common-defines"
JAVASCRIPT_HEADER = "javascript-header"
TEMP_BASE = system.getenv_text("TEMP_BASE",
                               tempfile.NamedTemporaryFile().name)
DEFAULT_CODE_CHECKERS = "jslint --maxerr 10000 --white,  jshint"
SAFEMODE_HEADER = """
'use strict';            // Added for sanity checking (e.g., undefined variables)
"""
DEFAULT_JAVASCRIPT_HEADER = """
// Start of added header (JavaScript and jQuery definitions)
function $(selector, context) { selector = context; }
var document;
var window;
var jQuery;
// End of added header
"""

class Script(Main):
    """Input processing class"""
    # TODO: -or-: """Adhoc script class (e.g., no I/O loop, just run calls)"""
    code_checkers = DEFAULT_CODE_CHECKERS
    strip_indent = False
    skip_safe_mode = False
    skip_common_defines = False
    javascript_header = DEFAULT_JAVASCRIPT_HEADER

    def __init__(self, *args, **kwargs):
        debug.trace_fmtd(5, "Script.__init__({a}): keywords={kw}; self={s}",
                         a=",".join(args), kw=kwargs, s=self)
        self.script_code = ""
        self.in_script = False
        self.code_indent = None
        super(Script, self).__init__(*args, **kwargs)

    def setup(self):
        """Check results of command line processing"""
        debug.trace_fmtd(5, "Script.setup(): self={s}", s=self)
        self.code_checkers = self.get_parsed_option(CODE_CHECKERS, self.code_checkers)
        self.strip_indent = self.get_parsed_option(STRIP_INDENT, self.strip_indent)
        self.skip_safe_mode = self.get_parsed_option(SKIP_SAFE_MODE, self.skip_safe_mode)
        self.skip_common_defines = self.get_parsed_option(SKIP_SAFE_MODE, self.skip_common_defines)
        self.javascript_header = self.get_parsed_option(JAVASCRIPT_HEADER, self.javascript_header)
        debug.trace_object(5, self, label="Script instance")

    def process_line(self, line):
        """Processes current line from input"""
        # Notes: Issues warning when src attribute given along with bracketted code
        # Also, issues warning when bracketted code on same line as <script> tag.
        debug.trace_fmtd(6, "Script.process_line({l})", l=line)
        entire_line = line
        script_tag_count = 0

        # Check for start of code section, ignoring external script via src attribute
        while my_re.search(r"^([^<>]*)<\/?script[^<>]*>(.*)", line):
            remainder = my_re.group(2)
            if my_re.search(r"^\s*<script[^<>]*>(.*)$", line):
                debug.assertion(not self.in_script)
                self.in_script = True
                line = my_re.group(1)
                script_tag_count += 1
            elif my_re.search(r"^(.*)</script[^<>]*>(.*)", line):
                debug.assertion(self.in_script)
                code = my_re.group(1).strip()
                line = my_re.group(2)
                self.in_script = False
                if code:
                    self.script_code += code + "\n"
            else:
                system.print_stderr("Warning: ignoring unexpected script tag formatting at line {n}: {t}", n=self.line_num, t=entire_line)
                line = remainder
        debug.assertion(script_tag_count <= 1)
        if script_tag_count:
            debug.trace_fmt(4, "{n} <script> taggings: in_script={ins}", n=script_tag_count, ins=self.in_script)

        # Accumulate text if within script tags
        if self.in_script:
            # Make sure indentation is defined if being stripped
            # Note: All code is stripped of indentation, so that aligns with javascript header.
            # This is just determined by first non-blank line, so that if-blocks, etc. not stripped of all spacing.
            if (self.strip_indent and line.strip()):
                # Make sure indentation defined
                if (self.code_indent is None):
                    self.code_indent = ""
                    if my_re.search(r"^(\s+)(.*)", line):
                        self.code_indent = my_re.group(1)
                        line = my_re.group(2)
                        debug.assertion(len(self.code_indent) > 0)
                    debug.trace_fmt(4, "Indent to strip: '{ind}' (len={l})", ind=self.code_indent, l=len(self.code_indent))
                # Strip indentation from start of line
                line = line.replace(self.code_indent, "", 1)
            #
            self.script_code += line.rstrip() + "\n"

    def wrap_up(self):
        """Run the accumulated script through code checkers"""
        if (not self.script_code.strip()):
            system.print_stderr("Error: No code found within <script> tags")
        javascript_file = (TEMP_BASE + ".js")

        # Add in header for strict mode (optional) and a few JavaScript defines
        code_header = ""
        if not self.skip_safe_mode:
            code_header += SAFEMODE_HEADER
        if not self.skip_common_defines:
            code_header += self.javascript_header
        system.write_file(javascript_file, code_header + self.script_code)
        output = None
        for checker in self.code_checkers.split(","):
            if output is not None:
                print("-" * 80)
            output = gh.run("{ch} {scr}", ch=checker, scr=javascript_file)
            print("Output from {ch}:".format(ch=checker))
            print(output)
            print("")

#-------------------------------------------------------------------------------
    
if __name__ == '__main__':
    debug.trace_current_context(level=debug.QUITE_DETAILED)
    app = Script(
        description=__doc__,
        # Note: skip_input controls the line-by-line processing, which is inefficient but simple to
        # understand; in contrast, manual_input controls iterator-based input (the opposite of both).
        skip_input=False,
        manual_input=False,
        # TODO: skip_input=True,
        # TODO: manual_input=True,
        boolean_options=[STRIP_INDENT, SKIP_SAFE_MODE, SKIP_COMMON_DEFINES],
        text_options=[
            (CODE_CHECKERS, "Comma-separated list of code checking invocations (e.g., '{dfc}')".format(dfc=DEFAULT_CODE_CHECKERS)),
            (JAVASCRIPT_HEADER, "JavaScript header with common definitions (e.g., document, window, jQuery)")])
    app.run()
