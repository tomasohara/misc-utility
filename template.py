#! /usr/bin/env python
#
# TODO what the file does (detailed)
#
# TODO:
# - add in common idioms (e.g. a la Python cheatsheet), for example:
#   class MyClass(MySuper):
#     def __init__(self, *args, **kwargs):
#        super(MyClass, self).__init__(*args, **kwargs)
#

"""TODO: what module does (brief)"""

import argparse
## TODO: import os
## TODO: import re
import sys
## TODO: from collections import defaultdict

import debug
import system

def main():
    """Entry point for script"""
    debug.trace_fmt(4, "main(): sys.argv={a}", a=sys.argv)

    # Check command-line arguments
    # TODO: add in detailed usage notes w/ environment option descriptions (see google_word2vec.py)
    parser = argparse.ArgumentParser(description="TODO: what the script does")
    # TODO:         formatter_class=argparse.RawDescriptionHelpFormatter
    #
    # TODO: env_options = system.formatted_environment_option_descriptions(indent="  ")
    # TODO: include expanded usage description
    #   description = """
    #     ...
    #     Notes:
    #     - ...
    #   """.format(...)

    #
    # TODO: use capitalized script description but lowercase argument help
    parser.add_argument("--TODO-bool-arg", default=False, action='store_true',
                        help="TODO: description")
    parser.add_argument("--TODO-int-arg", type=int, default=123,
                        help="TODO: description")
    parser.add_argument("--TODO-text-arg", default="TODO",
                        help="TODO: description")
    parser.add_argument("filename", nargs='?', default='-',
                        help="Input filename")
    args = vars(parser.parse_args())
    debug.trace_fmt(5, "args = {a}", a=args)
    filename = args['filename']
    # TODO: put supporting code in module to cut down on boilerplate code
    input_stream = sys.stdin
    if filename != "-":
        assert system.file_exists(filename)
        input_stream = open(filename, "r")
        assert input_stream
    else:
        debug.trace(6, "Processing stdin")

    # Process input line by line
    # TODO: implement the script
    line_num = 0
    for line in input_stream:
        line_num += 1
        line = line.strip("\n")
        debug.trace_fmt(5, "L{n}: {l}", n=line_num, l=line)

    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
