#! /usr/bin/env python
#
# randomize-lines.py: randomize lines in a file without reading entirely into memory.
# This creates a temporary file with a random numberr in the first column and
# the original line contents in the second. Then the temporary file is sorted
# and the random number column removed.
#
# Note:
# - inspired by examples under Stack Overflow (see below)  
#
#------------------------------------------------------------------------
# via http://stackoverflow.com/questions/4618298/randomly-mix-lines-of-3-million-line-file
#
# At the shell, use this.
#   python decorate.py | sort | python undecorate.py
#   
# decorate.py:
#   
#   import sys
#   import random
#   for line in sys.stdin:
#       sys.stdout.write( "{0}|{1}".format( random.random(), line ) )
#   
# undecorate.py:
#   
#   import sys
#   for line in sys.stdin:
#       _, _, data= line.partition("|")
#       sys.stdout.write( line )
#
#------------------------------------------------------------------------
# TODO;
# - Add sanity check for disk space issues.
# - Have streamlined version just using output from sort.
#

import argparse
import os
import random
import re
import sys
import tempfile

import tpo_common as tpo
import glue_helpers as gh

RANDOM_SEED = tpo.getenv_integer("RANDOM_SEED", None,
                                 "Integral seed for random number generation")

def main():
    """Entry point for script"""
    tpo.debug_print("main(): sys.argv=%s" % sys.argv, 4)

    # Check command-line arguments
    parser = argparse.ArgumentParser(description="Randomize lines in a file (without reading entirely into memory).")
    parser.add_argument("--include-header", default=False, action='store_true', help="random seed")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("filename", nargs='?', default='-', help="Input filename")
    args = vars(parser.parse_args())
    tpo.debug_print("args = %s" % args, 5)
    filename = args['filename']
    input_stream = sys.stdin
    if (filename != "-"):
        assert(os.path.exists(filename))
        input_stream = open(filename, "r")
        assert(input_stream)
    else:
        tpo.debug_print("Processing stdin", 6)
    global RANDOM_SEED
    if args['seed']:
        RANDOM_SEED = int(args['seed'])
    include_header = args['include_header']

    # Initialize seed for optional random number generator
    if RANDOM_SEED:
        random.seed(RANDOM_SEED)

    # Add column with random number to temporary file
    temp_base = tpo.getenv_text("TEMP_FILE", tempfile.NamedTemporaryFile().name)
    temp_input_file = temp_base + ".input"
    temp_output_file = temp_base + ".output"
    temp_input_handle = open(temp_input_file, "w")
    assert(temp_input_handle)
    #
    header = None
    line_num = 0
    for line in input_stream:
        line_num += 1
        line = line.strip("\n")
        tpo.debug_print("IL%d: %s" % (line_num, line), 6)
        if (line_num == 1) and include_header:
            header = line
        else:
            temp_input_handle.write("%s\t%s\n" % (random.random(), line))
    num_input_lines = line_num
    temp_input_handle.close()

    # Sort by random-number column (1) and then remove temporary column
    gh.run("sort -n < {in_file} | cut -f2- >| {out_file}",
           in_file=temp_input_file, out_file=temp_output_file)

    # Display result
    # TODO: send output of command above to stdout
    temp_output_handle = open(temp_output_file, "r")
    assert(temp_output_handle)
    line_num = 0
    IO_error = False
    if include_header and header:
        print(header)
        line_num += 1
        tpo.debug_print("HL%d: %s" % (line_num, header), 6)
    for line in temp_output_handle:
        line_num += 1
        line = line.strip("\n")
        tpo.debug_print("RL%d: %s" % (line_num, line), 6)
        try:
            print(line)
        except:
            IO_error = True
            tpo.debug_print("Exception printing line %d: %s" % (line_num, str(sys.exc_info())), 4)
            break
    num_output_lines = line_num
    tpo.debug_print("%s input and %d output lines" % (num_input_lines, num_output_lines), 4)
    gh.assertion((num_input_lines == num_output_lines) or IO_error)
    temp_output_handle.close()

    # Cleanup (e.g., removing temporary files)
    if not tpo.detailed_debugging():
        gh.run("rm -vfr {base}*", base=temp_base)
    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
