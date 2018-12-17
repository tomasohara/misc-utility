#! /usr/bin/env python
#
# merge_notes.py: merge textual note files based on timestamps
#------------------------------------------------------------------------
# Sample input:
#
# - [some-notes.txt]
#   Thu 13 Nov 14
#
#      wrote great linear sort
#
#   Sat 15 Nov 14
#
#       rewrote not-so-great n log n sort
#
# - [more-notes.txt]
#   Fri 14 Nov 14
#
#      ran unit tests
#
#
# Sample output:
#
#   Thurs 13 Nov 14
#
#      wrote great linear sort
#
#   Fri 14 Nov 14
#
#      ran unit tests
#
#   Sat 15 Nov 14
#
#       rewrote not-so-great n log n sort
#
#-------------------------------------------------------------------------------
# Notes:
# - Includes indicator of original file and line number in output
#      [vm-plata-notes.txt:44]
#      Did this and that
#
# Copyright (c) 2012-2018 Thomas P. O'Hara
#

"""Merge textual notes"""

import argparse
import datetime
import fileinput
import os
import re
import sys
from collections import defaultdict

import tpo_common as tpo
import glue_helpers as gh

def resolve_date(textual_date, default_date=None):
    """Converts from textual DATE into datetime object (using optional DEFAULT value)"""
    # EX: resolve_date("1 Jan 00") => datetime.datetime(2000, 1, 1, 0, 0)
    # Note: date component specifiers: %a: abbreviated weekday; %d day of month (2 digits); %b abbreviated month; %y year without century; %Y: year with century
    date = default_date
    resolved = False
    for date_format in ["%a %d %b %y", "%d %b %y", "%a %d %b %Y", "%d %b %Y"]:
        try:
            date = datetime.datetime.strptime(textual_date, date_format)
            resolved = True
            break
        except ValueError:
            pass
    if not resolved:
        tpo.debug_format("Unable to resolve date '{t}'", 2, t=textual_date)
    tpo.debug_format("resolve_date({t}, {d}) => {r}", 5,
                     t=textual_date, d=default_date, r=date)
    return date


def main():
    """Entry point for script"""
    tpo.debug_print("main(): sys.argv=%s" % sys.argv, 4)

    # Check command-line arguments
    parser = argparse.ArgumentParser(description="Merges ascii notes files")
    parser.add_argument("--ignore-dividers", default=False, action='store_true', help="Ignore lines consisting soley of dashes")
    parser.add_argument("--output-dividers", default=False, action='store_true', help="Output divider lines (80 dashes) betweens sections for different days")
    parser.add_argument("--show-file-info", default=False, action='store_true', help="Include filename and line number of original file in output")
    parser.add_argument("filename", nargs='+', default='-', help="Input filename")
    args = vars(parser.parse_args())
    tpo.debug_print("args = %s" % args, 5)
    input_files = args['filename']
    ignore_dividers = args['ignore_dividers']
    output_dividers = args['output_dividers']
    show_file_info = args['show_file_info']

    # Process input line by line
    # TODO: implement the script
    line_num = 0
    notes_hash = defaultdict(str)
    resolved_date = {}

    # Initiliaze current date to dummy from way back when
    dummy_date = "1 Jan 1900"
    resolved_dummy_date = resolve_date(dummy_date)
    gh.assertion(resolved_dummy_date < resolve_date("1 Jan 00"))
    last_date = dummy_date
    needs_source_info = True

    # Read in all the notes line by line
    for line in fileinput.input(input_files):
        line_num += 1
        line = line.strip("\n")
        tpo.debug_print("L%d: %s" % (line_num, line), 6)

        # Reset default date if first line in file
        if fileinput.isfirstline():
            tpo.debug_format("new file: {f}", 4, f=fileinput.filename())
            last_date = dummy_date
            needs_source_info = True
            line_num = 1

        # Optionally ignore section dividers (20 or more dashes)
        if (ignore_dividers and re.search("^--------------------+$", line)):
            tpo.debug_format("Ignoring divider at line {n}: {l}", 5, l=line, n=line_num)
            continue

        # Look for a new date in format Day dd Mon yy (e.g., "Fri 13 Nov 13")
        # Notes:
        # - Day and Mon are capitized 3-letter abbreviations (i.e.., Sun, ..., Sat and Jan, ..., Dec)
        # - Source file and line information will be added for each new date
        # TODO: allow for a variety of date formats; allow for optional time
        date = last_date
        if (re.search(r"^([a-z][a-z][a-z] )?\d+ [a-z][a-z][a-z] \d+$", line, re.IGNORECASE)):
            date = line.strip()
            needs_source_info = True

            # Resolve date format
            if date not in resolved_date:
                resolved_date[date] = resolve_date(date, last_date)
                # TODO: only add source info if different date
                # needs_source_info = True

            # Update current date
            # note: used for subsequent lines without date specifications
            last_date = date

            # Trace date resolution
            tpo.debug_format("New date at line {n}: raw={raw}; resolved={new}\n", 5, 
                             n=line_num, raw=date, new=resolved_date[date])

        # Add optional source indicator to current date
        if show_file_info and needs_source_info:
            notes_hash[date] += "[src={f}:{n}]\n".format(f=fileinput.filename(), 
                                                         n=fileinput.filelineno())
            needs_source_info = False
                
        # Add line to notes for current date
        gh.assertion(date != dummy_date)
        notes_hash[date] += line + "\n"
        

    # Sort the note entries by resolved date
    #
    # DEBUG
    def get_resolved_date(k):
        """Debugging accessor for resolved_date with tracing"""
        ## OLD: r = resolved_date.get(k)
        r = resolved_date.get(k, resolved_dummy_date)
        tpo.debug_format("get_resolved_date({k}) => {r}", 7, k=k, r=r)
        return r
    #
    tpo.debug_format("notes_hash keys: {{\n{k}\n}}", 7, 
                     k="\t\n".join([str(v) for v in notes_hash.keys()]))
    #
    for pos, date in enumerate(sorted(notes_hash.keys(), 
                                      ## OLD: key=lambda k: resolved_date.get(k, dummy_date))):
                                      ## BAD: key=lambda k: resolved_date.get(k))):
                                      ## DEBUG:
                                      key=get_resolved_date)):
        tpo.debug_format("outputting notes for date {d} [resolved: {r}]", 6, 
                         d=date, r=resolved_date.get(date))
        if output_dividers and (pos > 0):
            print("-" * 80)
        tpo.debug_format("[src={f}:{n}]", 6, skip_newline=True,
                         f=fileinput.filename(), n=fileinput.filelineno())
        print("%s\n\n" % notes_hash[date])

    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
