#! /usr/bin/python
#
# parse-text.py: run text through grammatial parser. This currently just supports the Stanford Parser
# (version 2013-04-05), but other parsers can be added by specifying the command-line sequence and 
# adding code to isolate the parse tree from other ouput by the parser.
#
# The input is a text files with sentences separated by blank lines. After each sentence is parsed,
# the parse tree can optionally be displayed via NLTK. Under Cygwin, this requires an active X-Server.
#
# Sample input:
#    This be me car over over there.
#
# Sample output:
#    (ROOT
#      (S
#        (NP (DT This))
#        (VP (VB be)
#          (NP
#            (NP (PRP me) (NN car))
#            (PP (IN over)
#              (PP (IN over)
#                (NP (RB there))))))
#        (. .)))
#
#------------------------------------------------------------------------
# Complete output from parser:
#
#    $ run-stanford-parser.sh badder-grammar-example.txt
#    Loading parser from serialized file edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz ... done [9.4 sec].
#    Parsing file: D:\cartera-de-tomas\apeters-essay-grading\badder-grammar-example.txt
#    Parsing [sent. 1 len. 8]: This be me car over over there .
#    (ROOT
#      (S
#        (NP (DT This))
#        (VP (VB be)
#          (NP
#            (NP (PRP me) (NN car))
#            (PP (IN over)
#              (PP (IN over)
#                (NP (RB there))))))
#        (. .)))
#    
#    Parsing [sent. 2 len. 4]: You seen it ?
#    (ROOT
#      (S
#        (NP (PRP You))
#        (VP (VBN seen)
#          (NP (PRP it)))
#        (. ?)))
#    
#    Parsed file: D:\cartera-de-tomas\apeters-essay-grading\badder-grammar-example.txt [2 sentences].
#    Parsed 12 words in 2 sentences (17.49 wds/sec; 2.92 sents/sec).
#    
# ------------------------------------------------------------------------
# Notes:
# - The STANFORD_PARSER_HOME environment variable specified the distribution
# directory for the parser. See run-stanford-parser.sh.
#

#------------------------------------------------------------------------
# Library packages

from common import *

# OS support
import sys
import commands
import tempfile

#------------------------------------------------------------------------
# Globals

# Whether to use NLTK to draw the parse tree(s)
GRAPH_PARSE_TREE = getenv_boolean("GRAPH_PARSE_TREE", False)

#------------------------------------------------------------------------
# Optional libraries

if GRAPH_PARSE_TREE:
    import nltk

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

def parse_file(file):
    """Run grammar checker over FILE, returning result as a string. Note that mutiple parse trees with be separated by a form feed character (i.e., \f)"""
    # Get command output
    script_dir = os.path.dirname(__file__)
    command_line = "%s/run-stanford-parser.sh %s" % (script_dir, file)
    result = run_command(command_line)

    # Extract parse tree from result (see complete output sample above in header comments)
    parse_tree_output = ""
    lines = result.split("\n")
    in_parse = False
    for line_num in range(1, len(lines)):
        # Note: only strips from right to preserve indentation (stripping need for blank line check under Cygwin)
        line = lines[line_num - 1].rstrip()
        debug_print("Parser L%d: %s" % (line_num, line), 5)

        # Scan for parse tree section (between 'Parsing [sent...] and blank line)
        if (re.search("Parsing\s*\[sent", line)):
            in_parse = True
            current_parse_tree = ""
        elif in_parse:
            if (line == ""):
                in_parse = False
                # Add to master output and optionally graph parse tree
                if (len(parse_tree_output) > 0):
                    parse_tree_output += "\f"
                parse_tree_output += current_parse_tree
                if GRAPH_PARSE_TREE:
                    tree = nltk.tree.Tree.parse(current_parse_tree)
                    nltk.draw.tree.draw_trees(tree)
                    print_stderr("Close NLTK window to continue.")
            else:
                current_parse_tree += line + "\n"
        elif (re.search("^\s*\(", line)):
            debug_print("Warning: potential parse tree segment missed at line %d: %s" % (line_num, line))
        else:
            debug_print("Ignoring parser output line %d: %s" % (line_num, line), 4)

    # Return string with all parse trees, warning if none found
    if (parse_tree_output == ""):
        debug_print("Warning: unable to extract parse tree")                         
    return parse_tree_output

def parse_text(text):
    """Run parser over TEXT"""
    temp_file=tempfile.NamedTemporaryFile()
    debug_print("temp_file: %s" % temp_file.name, 5)
    with open(temp_file.name, 'w') as f:
        f.write(text)
    return parse_file(temp_file.name)

def main():
    """
    Main routine: parse arguments and perform grammatical parsing of file or text.
    Note: main() used to avoid conflicts with globals (e.g., if code inline at end of script).
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
        print_stderr(" - set STANFORD_PARSER_HOME environment variable to directory for distribution (see http://nlp.stanford.edu/software/lex-parser.shtml)")
        print_stderr("- set GRAPH_PARSE_TREE environment variable to 1 to graph each parser tree via NLTK")
        sys.exit()
    arg_pos = 1
    while (arg_pos < num_args) and (args[arg_pos][0] == "-"):
        debug_print("args[%d]: %s" % (arg_pos, args[arg_pos]), 3)
        if (args[arg_pos] == "-"):
            # note: - is used to avoid usage statement with file input from stdin
            pass
        elif (args[arg_pos] == "--text"):
            arg_pos += 1
            print("%s" % parse_text(args[arg_pos]))
            sys.exit()
        else:
            print_stderr("Error: unexpected argument '%s'" % args[arg_pos])
            sys.exit()
        arg_pos += 1
    file = args[arg_pos]

    # Get parse tree for text and optionally draw it
    print("%s" % parse_file(file))

    # Cleanup
    debug_print("stop %s: %s" % (__file__, debug_timestamp()), 3)

#------------------------------------------------------------------------
# Standalone processing

if __name__ == '__main__':
    main()
