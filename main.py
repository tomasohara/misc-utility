#! /usr/bin/env python
#
# Provides class Main to encompass common script processing. By default, the
# command line arguments are analyzed to determine optional filename, which is
# opened. Then, the input stream is feed line-by-line into the process_line
# method.
#
# Usage exmple:
#    from main import Main
#
#    class MyMain(Main):
#       def process_line(self, line):
#           if "funny" in line:
#               print("Funny looking line: %s" % line)
#
#    if __name__ == '__main__':
#        MyMain().run()
#    
# Notes:
# - See simple_main_example.py for a non-trivial example.
# - To add command-line arguments, pass corresponding arguments to Main's
#   initialization. For example,
#      Main(boolean_options=["fubar"], 
#           int_options=[("count", "Number of times", 10)]).run()
# - As the class likely will just be instantiated once, initialization
#   can be simplfied by using class-level variables for options, as follows:
#      Script(Main):
#          count = 5
#          verbose = False
#          def setup(self):
#              verbose = self.get_parsed_option("verbose")
# - With non-trivial command processing (e.g., positional arguments), it 
#   might be better to do this in the constructor, as follows:
#       def __init__(*args, **kwargs):
#           super(MyMain, self).__init__(*args, positional_options=["targets"], 
#                                        **kwargs)
# - Changes to temporary directory/file support should be synchronized with the
#   unit testing base class (see tests/unittest_wrapper.py.
# - Overriding the temporary directory can be handy during debugging (via
#   TEMP_BASE or TEMP_FILE). If you invoke sub-scripts, you might need to
#   specify different ones, as in adhoc/optimize_company_extraction.py.
#
# TODO:
# - Specify argument via input dicts, such as in 
#      options=[{"name"="verbose", "type"=bool}, 
#               {"name"="count", type=int, default=10}]
# - Add support for perl-style paragraph mode in input processing.
# - Add support for multple input files (e.g., via fileinput module).
# - Add support for csv.csv_reader (see usage in cut.py).
#

"""Module for encapsulating main() processing"""

import argparse
import os
import sys
import tempfile

import debug
import tpo_common as tpo
import glue_helpers as gh

class Main(object):
    """Class encompassing common script processing"""
    argument_parser = None
    force_unicode = False

    def __init__(self, runtime_args=None, description=None, 
                 # TODO: Either rename xyz_optiom to match python type name 
                 # or rename them without abbreviations.
                 # TODO: explain difference between positional_options and positional_arguments
                 boolean_options=None, text_options=None, int_options=None,
                 float_options=None, positional_options=None, positional_arguments=None, 
                 skip_input=None, manual_input=None, auto_help=None):
        """Class constructor: parses RUNTIME_ARGS (or command line), with specifications
        for BOOLEAN_OPTIONS, TEXT_OPTIONS, INT_OPTIONS, FLOAT_OPTIONS, and POSITIONAL_OPTIONS
        (see convert_option). Includes options to SKIP_INPUT, or to have MANUAL_INPUT, or to use AUTO_HELP invocation (i.e., assuming --help if no args)."""
        tpo.debug_format("Main.__init__({args}, d={desc}, b={bools}, t={texts}, "
                         + "i={ints}, f={floats}, p={posns}, s={skip}, m={mi}, a={auto})", 5,
                         args=runtime_args, desc=description, bools=boolean_options,
                         texts=text_options, ints=int_options, floats=float_options,
                         posns=positional_options, skip=skip_input, mi=manual_input, auto=auto_help)
        self.description = "TODO: what the script does" # defautls to TODO note for client
        # TODO: boolean_options = [(VERBOSE, "Verbose output mode")]
        self.boolean_options = []
        self.text_options = []
        self.int_options = []
        self.float_options = []
        self.positional_options = []
        self.process_line_warning = False
        self.input_stream = None
        self.line_num = -1
        # Note: manual_input was introduced after skip_input to allow for input processing
        # in bulk (e.g., via read_input generator). By default, neither is specified
        # (see new_template.py), and both should be assumed false.
        # TODO: *** Add better sanity checking (such as a filename on command line).
        if manual_input is None:
            ## BAD1: manual_input = (skip_input is not None)
            ## BAD2: manual_input = False
            ## NOTE: skip_input=>manual_input: T=>T  F=>F  None=>F
            manual_input = False if (skip_input is None) else skip_input
            debug.trace_fmt(7, "inferred manual_input: {mi}", mi=manual_input)
        self.manual_input = manual_input
        if skip_input is None:
            ## BAD: skip_input = (not self.manual_input)
            skip_input = self.manual_input
            debug.trace_fmt(7, "inferred skip_input: {si}", si=skip_input)
        self.skip_input = skip_input
        #
        self.parser = None
        if auto_help is None:
            auto_help = self.skip_input
        self.auto_help = auto_help

        # Setup temporary file and/or base directory
        self.temp_base = tpo.getenv_text("TEMP_BASE",
                                         tempfile.NamedTemporaryFile().name)
        # TODO: self.use_temp_base_dir = gh.dir_exists(gh.basename(self.temp_base))
        # -or-: temp_base_dir = tpo.getenv_text("TEMP_BASE_DIR", ""); self.use_temp_base_dir = bool(temp_base_dir.strip); ...
        self.use_temp_base_dir = tpo.getenv_bool("USE_TEMP_BASE_DIR", False)
        if self.use_temp_base_dir:
            gh.run("mkdir -p {dir}", dir=self.temp_base)
            default_temp_file = gh.form_path(self.temp_base, "temp.txt")
        else:
            default_temp_file = self.temp_base
        self.temp_file = tpo.getenv_text("TEMP_FILE", default_temp_file)

        # Get arguments from specified parameter or via command line
        # Note: --help assumed for input-less scripts with command line options
        # to avoid inadvertant script processing.
        if runtime_args is None:
            runtime_args = sys.argv[1:]
            tpo.debug_print("Using sys.argv for runtime args: %s" % runtime_args, 4)
            if self.auto_help and not runtime_args:
                runtime_args = ["--help"]
        # Get other options
        if description:
            self.description = description
        if boolean_options:
            self.boolean_options = boolean_options
        if text_options:
            self.text_options = text_options
        if int_options:
            self.int_options = int_options
        if float_options:
            self.float_options = float_options
        if positional_options or positional_arguments:
            # TODO: mark positional_options as decprecated
            gh.assertion(not (positional_options and positional_arguments))
            self.positional_options = positional_options or positional_arguments
        # Set defaults
        self.parsed_args = None
        self.filename = None
        # Do command-line parsing
        self.check_arguments(runtime_args)
        debug.trace_current_context(level=debug.QUITE_DETAILED)
        debug.trace_object(6, self, label="Main instance")
        debug.trace_fmt(tpo.QUITE_DETAILED, "end of Main.__init__(); self={s}",
                        s=self)
        return

    def convert_option(self, option_spec, default_value=None, positional=False):
        """Convert OPTION_SPEC to (label, description, default) tuple. 
        Notes: The description and default of the specification are optional,
        and the parentheses can be omitted if just the label is given. Also,
        if POSITIONAL the option prefix (--) is omitted."""
        opt_label = None
        opt_desc = None
        opt_default = default_value
        opt_prefix = "--" if not positional else ""
        if isinstance(option_spec, tuple):
            option_components = list(option_spec)
            opt_label = opt_prefix + option_components[0]
            if len(option_components) > 1:
                opt_desc = option_components[1]
            if len(option_components) > 2:
                opt_default = option_components[2]
        else:
            opt_label = opt_prefix + tpo.to_string(option_spec)
        gh.assertion(not " " in opt_label)
        result = (opt_label, opt_desc, opt_default)
        tpo.debug_format("convert_option({o}, {d}, {p}): self={s} => {r}", 5,
                         o=option_spec, d=default_value, p=positional,
                         s=self, r=result)
        return result

    def get_option_name(self, label):
        """Return internal name for parser options (e.g. dashes converted to underscores)"""
        name = label.replace("-", "_")
        tpo.debug_format("get_option_name({l}) => {n}; self={s}", 6,
                         l=label, n=name, s=self)
        return name

    def has_parsed_option(self, label):
        """Whether option for LABEL specified (i.e., non-null value)"""
        name = self.get_option_name(label)
        has_option = (name in self.parsed_args and self.parsed_args[name])
        tpo.debug_format("has_parsed_option({l}) => {r}", 6,
                         l=label, r=has_option)
        return has_option

    def get_parsed_option(self, label, default=None, positional=False):
        """Get value for option LABEL, with dashes converted to underscores. 
        If POSITIONAL specified, DEFAULT value is used if omitted"""
        opt_label = self.get_option_name(label) if not positional else label
        value = self.parsed_args.get(opt_label)
        # Override null value with default
        if value is None:
            value = default
            # Do sanity check for positional argument being checked by mistake
            # TODO: do automatic correction?
            if opt_label != label:
                if positional:
                    gh.assertion(opt_label not in self.parsed_args)
                else:
                    gh.assertion(label not in self.parsed_args)
        # Return result, after tracing invocation
        tpo.debug_format("get_parsed_option({l}, [{d}], [{p}]) => {v}", 5,
                         l=label, d=default, p=positional, v=value)
        return value

    def get_parsed_argument(self, label, default=None):
        """Get value for positional argument LABEL using DEFAULT value"""
        tpo.debug_format("get_parsed_agument({l}, [{d}])", 6,
                         l=label, d=default)
        return self.get_parsed_option(label, default, positional=True)

    def check_arguments(self, runtime_args):
        """Check command-line arguments"""
        tpo.debug_format("Main.check_arguments({args})", 5, args=runtime_args)
        # TODO: add in detailed usage notes w/ environment option descriptions (see google_word2vec.py)
        if not self.argument_parser:
            self.argument_parser = argparse.ArgumentParser
        parser = self.argument_parser(description=self.description)
        # TODO: use capitalized script description but lowercase argument help

        # Check for options of specific types
        # TODO: consolidate processing for the groups; add option for environment-based default
        for opt_spec in self.boolean_options:
            (opt_label, opt_desc, opt_default) = self.convert_option(opt_spec, None)
            parser.add_argument(opt_label, default=opt_default, action='store_true',
                                help=opt_desc)
        for opt_spec in self.int_options:
            (opt_label, opt_desc, opt_default) = self.convert_option(opt_spec, None)
            parser.add_argument(opt_label, type=int, default=opt_default, help=opt_desc)
        for opt_spec in self.float_options:
            (opt_label, opt_desc, opt_default) = self.convert_option(opt_spec, None)
            parser.add_argument(opt_label, type=float, default=opt_default,
                                help=opt_desc)
        for opt_spec in self.text_options:
            (opt_label, opt_desc, opt_default) = self.convert_option(opt_spec, None)
            parser.add_argument(opt_label, default=opt_default, help=opt_desc)

        # Add dummy arguments
        if tpo.detailed_debugging():
            if not self.boolean_options:
                parser.add_argument("--TODO-bool-arg", default=False, action='store_true',
                                    help="Add via boolean_options keyword")
            if not self.text_options:
                parser.add_argument("--TODO-text-arg", default="",
                                    help="Add via text_options keyword")
            if not self.int_options:
                parser.add_argument("--TODO-int-arg", type=int, default=0,
                                    help="Add via int_options keyword")
            if not self.float_options:
                parser.add_argument("--TODO-float-arg", default=0.0,
                                    help="Add via float_options keyword")

        # Add positional arguments
        for i, opt_spec in enumerate(self.positional_options):
            (opt_label, opt_desc, opt_default) = self.convert_option(opt_spec, "",
                                                                     positional=True)
            # note: a numeric nargs produces a list even if 1, so None used
            nargs = None
            tpo.debug_format("positional arg {i}, nargs={nargs}", 6, 
                             i=i, nargs=nargs)
            parser.add_argument(opt_label, default=opt_default, nargs=nargs, 
                                help=opt_desc)
        # Add filename last and make optional with '-' default (stdin)
        if not self.skip_input:
            filename_nargs = '?'
            tpo.debug_format("filename_nargs={nargs}", 6, nargs=filename_nargs)
            parser.add_argument("filename", nargs=filename_nargs, default='-',
                                help="Input filename")
        # Parse the commandline and get result
        tpo.debug_format("parser={p}", 6, p=parser)
        self.parser = parser
        self.parsed_args = vars(parser.parse_args(runtime_args))
        tpo.debug_print("parsed_args = %s" % self.parsed_args, 5)
        if not self.skip_input:
            self.filename = self.parsed_args['filename']
        return

    def setup(self):
        """Perform script setup prior to input processing"""
        # Note: Use for post-argument proceessing setup
        tpo.debug_format("Main.setup() stub: self={s}", 5, s=self)
        return

    def process_line(self, line):
        """Stub for input processing that just prints the input.
        Note: issues error message about required specialization"""
        tpo.debug_format("Main.process_line({l})", 5, l=line)
        if not self.process_line_warning:
            tpo.print_stderr("Internal error: specialize process_line")
            self.process_line_warning = True
        print(line)
        return

    def run_main_step(self):
        """Stub for main processing, along with error message"""
        # TODO: use decorator (e.g., @abstract)
        tpo.debug_format("Main.run_main_step(): self={s}", 5, s=self)
        tpo.print_stderr("Internal error: specialize run_main_step")
        return

    def run(self):
        """Entry point for script"""
        tpo.debug_print("Main.run()", 5)
        # TODO: decompose (e.g., isolate input proecessing)

        # Have client do pre-input initialization
        self.setup()

        # Resolve input stream from either explicit filename or via standard input
        self.input_stream = sys.stdin
        if self.filename and (self.filename != "-"):
            gh.assertion(os.path.exists(self.filename))
            self.input_stream = open(self.filename, "r")
            gh.assertion(self.input_stream)
    
        # If not automatic input, process the main step of script
        if self.manual_input:
            self.run_main_step()
        # Otherwise have client process input line by line
        else:
            self.process_input()

        # Invoke client end processing
        self.wrap_up()

        # Remove any temporary files
        self.clean_up()
        return

    def read_input(self):
        """Generator for producing lines of text from the input
        Note: For use with self.manual_input"""
        tpo.debug_format("Main.read_input(): {input}", 5,
                         input=self.input_stream)
        self.line_num = 0
        for line in self.input_stream:
            self.line_num += 1
            line = line.strip("\n")
            tpo.debug_print("L%d: %s" % (self.line_num, line), 6)
            if self.force_unicode:
                line = tpo.ensure_unicode(line)
            tpo.debug_print("\ttype(line): %s" % (type(line)), 7)
            yield line
        return
    
    def process_input(self):
        """Process each line in current input stream (or stdin)"""
        tpo.debug_format("Main.process_input(): {input}", 5,
                         input=self.input_stream)
        self.line_num = 0
        for line in self.read_input():
            self.process_line(line)
        return

    def wrap_up(self):
        """Default end processing"""
        tpo.debug_format("Main.wrap_up() stub: self={s}", 5, s=self)
        return

    def clean_up(self):
        """Removes temporary files, etc."""
        # note: not intended to be overridden
        tpo.debug_format("Main.clean_up(): self={s}", 5, s=self)
        if not tpo.detailed_debugging():
            if self.use_temp_base_dir:
                gh.run("rm -rvf {dir}", dir=self.temp_base)
            else:
                gh.run("rm -vf {file}*", file=self.temp_file)
        return

#------------------------------------------------------------------------

if __name__ == '__main__':
    tpo.print_stderr("Warning: %s is not intended to be run standalone" % __file__)
