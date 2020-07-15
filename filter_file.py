#! /usr/bin/env python
#
# Extracts portion of a file, based on occurence of tokens in another file. The arguments consists
# of a file listing the cases defining the filter along with a regex for extracting the candidate
# cases from the input. For example, the filter file could be a list of names, and the regex the
# pattern 'Hello (.*)'. Then, all lines matching Hello with one of the names will be output.
#
# Note:
# - See simple_main_example.py for a simple regex filter.
#
# TODO:
# - Show usage examples.
# - Add fitlering based on field number rather than regex.
# - Rework in terms of re.search.
#

import argparse
import re
import sys
import tpo_common as tpo
from tpo_common import debug_print
from glue_helpers import read_lines

DEFAULT_REGEX="^(.*)$"

def main():
    """Entry point for script"""
    # Check command-line arguments
    parser = argparse.ArgumentParser(description=
"""
Filters text from input based on occurrence of text in another file (i.e., cases that define
filter). This is intended to handle filters with numerous cases, making use of grep-based
filtering impractical.

Example:

filter_file.py  --regex='^(.*) =>.*' --exclude  bad-misspelling-cases.list  < misspelling-rewrites.txt  >  new-misspelling-rewrites.txt
""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regex", default=DEFAULT_REGEX, help="Regex with group 1 placeholder used to match filter")
    parser.add_argument("--field-num", type=int, default=0, help="Field number for tabular data")
    parser.add_argument("--include", dest='include', action='store_true', default=True, help="Whether to exclude matches (default)")
    parser.add_argument("--exclude", dest='include', action='store_false', help="Whether to exclude matches (rather than default inclusion)")
    parser.add_argument("--filter-cases", help="Specific cases to filter (comma separated list)")
    parser.add_argument("text-filter-file", help="File with cases that define filter")
    args = parser.parse_args()
    option = vars(args)
    debug_print("option: %s" % option, 4)
    include_matches = option['include']
    filter_regex = option['regex']
    text_filter_file = option['text-filter-file']
    field_num = option['field_num']
    assert ((field_num == 0) or (filter_regex == DEFAULT_REGEX)), "Field number and regex args are mutually exclusive"
    is_matching_case = {}
    if option['filter_cases']:
        for case in re.split(", *", option['filter_cases']):
            is_matching_case[case] = True

    # Read list of cases for filtering input
    if (text_filter_file != "-"):
        for line in read_lines(text_filter_file):
            debug_print("FL%d: '%s'" % ((1 + len(is_matching_case)), line), 7)
            is_matching_case[line] = True
    debug_print("len(is_matching_case) = %d" % len(is_matching_case), 4)
    debug_print("is_matching_case = %s" % is_matching_case, 8)

    # Output portions of input passing filter
    line_num = 0
    for line in sys.stdin:
        line = line.strip("\n")
        line_num += 1
        debug_print("L%d: %s" % (line_num, line), 6)

        # Extract filter target from input
        text = None
        if (field_num > 0):
            fields = line.split("\t")
            tpo.trace_array(fields, 8, "fields")
            ## tpo.debug_print("fields=%s" % fields, 8)
            ## tpo.trace_object(fields, 8, "fields")
            if field_num <= len(fields):
                text = fields[field_num - 1]
        else:
            debug_print("trying re.match", 8)
            match = re.match(filter_regex, line)
            if match:
                text = match.group(1)
        debug_print("Checking text '%s' against filter" % text, 7)

        # see if matches filtered cases
        matches_filter = False
        if ((text is not None) and (text in is_matching_case)):
            matches_filter = True
            debug_print("Line %d matches case %s of filter file: %s" % (line_num, text, line), 5)

        # Print if one of the cases from the hash matched (or if no matches in hash for exclusion filters)
        include_line = (matches_filter == include_matches)
        debug_print("matches_filter=%s include_matches=%s include_line=%s" % (matches_filter, include_matches, include_line), 7)
        if include_line:
            print(line)
    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
