#! /usr/bin/python
#
# check-grammar.py: run text through grammar checker (e.g., the one from OpenOffice)
#
# Sample input:
#    This be me car over over there
#
# Sample output:
#    Line 1, column 16, Rule ID: ENGLISH_WORD_REPEAT_RULE
#    Message: Possible typo: you repeated a word
#    Suggestion: over
#    This be me car over over there 
#                   ^^^^^^^^^
#
# ------------------------------------------------------------------------
# Notes:
# - The LANGUAGE_TOOL_HOME environment variable specified the distribution
# directory for the grammar checker. See run-languagetool-grammar-checker.sh.
# - Underylying support downloaded from
#   https://extensions.openoffice.org/en/projectrelease/languagetool-21
#
#-------------------------------------------------------------------------------
# TODO:
# - ???
#

#------------------------------------------------------------------------
# Library packages

from common import *

# OS support
import os
import sys
import commands
import tempfile

#------------------------------------------------------------------------
# Functions

def run_command(command_line, level=5):
    """Runs COMMAND_LINE and returns the output (as a string), with debug tracing at specified LEVEL"""
    # Issue command
    debug_print("Running command: %s" % command_line, level=level)
    (status, output) = commands.getstatusoutput(command_line)
    if (status != 0):
        print_stderr("Warning: problem running command (status=%d): %s" % (status, command_line))

    # Return output
    debug_print("Command output: %s\n" % output, level=level+1)
    return (output)

def check_file_grammar(file):
    """Run grammar checker over FILE"""
    # Get command output
    script_dir = os.path.dirname(__file__)
    command_line = "%s/run-languagetool-grammar-checker.sh %s" % (script_dir, file)
    result = run_command(command_line)

    # Remove trace messages output
    # ex: "No language specified, using English\nWorking on D:\cartera-de-tomas\apeters-essay-grading\bad-grammar-example.txt...\n"
    # ex: "Time: 171ms for 1 sentences (5.8 sentences/sec)"
    result = re.sub(r"No language specified.*", "", result)
    result = re.sub(r"Working on.*", "", result)
    result = re.sub(r"Time:.*", "", result)

    return result

def check_text_grammar(text):
    """Run grammar checker over TEXT"""
    temp_file=tempfile.NamedTemporaryFile()
    debug_print("temp_file: %s" % temp_file.name, 3)
    with open(temp_file.name, 'w') as f:
        f.write(text)
    return check_file_grammar(temp_file.name)

def main():
    """
    Main routine: parse arguments and perform grammar checking over file or text.
    Note: main() Used to avoid conflicts with globals (e.g., if done at end of script).
    """
    # Init
    debug_print("start %s: %s" % (__file__, debug_timestamp()), 3)

    # Parse command-line, showing usage statement if no arguments given (or --help)
    args = sys.argv
    debug_print("argv = %s" % sys.argv, 3)
    num_args = len(args)
    if ((num_args == 1) or ((num_args > 1) and (args[1] == "--help"))):
        print_stderr("Usage: %s [--text text] [--help] file" % args[0])
        print_stderr("")
        print_stderr("Notes:")
        print_stderr(" - set LANGUAGE_TOOL_HOME environment variable to directory for LanguageTool distribution (available via www.languagetool.org)")
        sys.exit()
    arg_pos = 1
    while (arg_pos < num_args) and (args[arg_pos][0] == "-"):
        debug_print("args[%d]: %s" % (arg_pos, args[arg_pos]), 3)
        if (args[arg_pos] == "-"):
            # note: - is used to avoid usage statement with file input from stdin
            pass
        elif (args[arg_pos] == "--text"):
            arg_pos += 1
            # Run grammar checker over sentence from command line
            print("%s" % check_text_grammar(args[arg_pos]))
            sys.exit()
        else:
            print_stderr("Error: unexpected argument '%s'" % args[arg_pos])
            sys.exit()
        arg_pos += 1
    file = args[arg_pos]

    # Run grammar checker over sentences in file
    print("%s" % check_file_grammar(file))

    # Cleanup
    debug_print("stop %s: %s" % (__file__, debug_timestamp()), 3)

#------------------------------------------------------------------------
# Standalone processing

if __name__ == '__main__':
    main()
