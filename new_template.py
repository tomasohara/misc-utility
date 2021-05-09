#! /usr/bin/env python
# 
# TODO what the script does (detailed)
#

"""TODO: what module does (brief)"""

from main import Main
import debug
# TODO: from regex import my_re
import system

## TODO: Constants for switches omitting leading dashes (e.g., DEBUG_MODE = "debug-mode")
## Note: Run following in Emacs to interactively replace TODO_ARGn with option label
##    M-: (query-replace-regexp "todo\\([-_]\\)argn" "arg\\1name")
## where M-: is the emacs keystroke short-cut for eval-expression.
TODO_ARG1 = False
TODO_ARG1 = "TODO-arg1"
TODO_ARG2 = "TODO-arg2"
## TODO_FILENAME = "TODO-filename"

class Script(Main):
    """Input processing class"""
    # TODO: -or-: """Adhoc script class (e.g., no I/O loop, just run calls)"""
    TODO_arg1 = False
    TODO_arg2 = ""

    # TODO: add class constructor
    ## def __init__(self, *args, **kwargs):
    ##     debug.trace_fmtd(5, "Script.__init__({a}): keywords={kw}; self={s}",
    ##                      a=",".join(args), kw=kwargs, s=self)
    ##     super(Script, self).__init__(*args, **kwargs)
    
    def setup(self):
        """Check results of command line processing"""
        debug.trace_fmtd(5, "Script.setup(): self={s}", s=self)
        self.TODO_arg1 = self.get_parsed_option(TODO_ARG1, self.TODO_arg1)
        self.TODO_arg2 = self.get_parsed_option(TODO_ARG2, self.TODO_arg2)
        # TODO: self.TODO_filename = self.get_parsed_argument(TODO_FILENAME)
        debug.trace_object(5, self, label="Script instance")

    def process_line(self, line):
        """Processes current line from input"""
        debug.trace_fmtd(6, "Script.process_line({l})", l=line)
        # TODO: flesh out
        if self.TODO_arg1 and "TODO" in line:
            print("arg1 line: %s" % line)
        ## TODO: regex pattern matching
        ## elif my_re.search(self.TODO_arg2, line):
        ##     print("arg2 line: %s" % line)

    ## TODO: if no input proocessed, customize run_main_step instead
    ## and specify skip_input below
    ##
    ## def run_main_step(self):
    ##     """Main processing step"""
    ##     debug.trace_fmtd(5, "Script.run_main_step(): self={s}", s=self)
    ##     # ...
    ##     output = gh.run("program {opts} < {input}", opts="...", input="...")
    ##     # ...

    ## TODO: def wrap_up(self):
    ##           # ...

    ## TODO: def clean_up(self):
    ##           # ...
    ##           super(Script, self).clean_up()

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
        boolean_options=[TODO_ARG1],
        # TODO: positional_options=[TODO_FILENAME],
        text_options=[(TODO_ARG2, "TODO-desc")])
    app.run()
