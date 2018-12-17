#! /usr/bin/env python
#
# Sanity check for time-tracking report.
#
# Sample input:
#
# Week of Mon 14 Oct 14
# 
# Mon:
# ;; day off
# hours: 0
# 
# Tues:
# 2	3	appserver overview by Aaron; diagnosing hg update issues on juju repository
# 5	6	re-cloning and reviewing juju repository source structure
# 7	8	adding new repository branch; checking old juju repository differences (e.g., TODO notes in code)
# 9:30	11:30	exporting related-title code from juju_prototype into appserver; adding prequisites for running appserver locally (for syntax checking)
# 12	2	trying to get @route-based dispatching for relatedJobs method
# hours: 7
#
# TODO:
# - Allow _ placeholders in week and total summary if month not complete.
# - Clean up debug_format calls.
# - Make hours line optional (e.g., assume blank line separates days).
# - Make weekly hours optional (e.g., assume repeated day separates week).
# - Have option to output template for current month!
#
# Copyright (c) 2012-2018 Thomas P. O'Hara
#

"""Validates the hour tabulations in a time tracking report"""

import argparse
import os
import re
import sys

import debug
import tpo_common as tpo
from regex import my_re

WEEKDAYS = ["sun", "mon", "tues", "wed", "thurs", "fri", "sat",
            "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]

def main():
    """Entry point for script"""
    tpo.debug_print("main(): sys.argv=%s" % sys.argv, 4)

    # Check command-line arguments
    parser = argparse.ArgumentParser(description="Validates time tracking reports")
    parser.add_argument("--strict", default=False, action='store_true', help="strict parsing of the time tracking report")
    parser.add_argument("--ignore-comments", default=False, action='store_true', help="ignore comments in the time tracking report")
    parser.add_argument("--verbose", default=True, action='store_true', help="verbose output mode (e.g., summary of weekly hours)")
    parser.add_argument("--quiet", dest='verbose', action='store_false', help="non-verbose mode")
    parser.add_argument("filename", help="input filename")
    args = vars(parser.parse_args())
    tpo.debug_print("args = %s" % args, 5)
    filename = args['filename']
    assert(os.path.exists(filename))
    strict = args['strict']
    ignore_comments = args['ignore_comments'] or (not strict)
    verbose = args['verbose']

    # Scan time listing keeping track of hours running totals
    hours = 0
    weekly_hours = 0
    total_hours = 0
    line_num = 0
    num_weeks = 0
    for line in open(filename):
        line = line.strip("\n")
        line_num += 1
        tpo.debug_format("L{line_num}: {line}", 6)

        # Normalize line to facilitate extraction:
        # - treat whitespace delimiters as tabs after start and end time
        # - remove surrounding whitespace, make lowercase
        # - remove am/pm suffixes in time specifications
        line = line.lower().strip()
        if (re.search("^(\d+\S*)\s+(\d+\S*)\s+", line)) :
            line = re.sub(r"^(\S+)\s+(\S+)\s+", r"\1\t\2\t", line)
            debug.assertion(not line.endswith("\t"))
            debug.assertion(not re.search("\t.*\t.*\t",line))
            line = re.sub(r"(am|pm)\t", r"\t", line)
        tpo.debug_format("Normalized L{line_num}: {line}", 5)

        # Ignore comments (line starting either with '#' or ';' ignoring leading space)
        if (ignore_comments and re.search(r"^\s*[#;]", line)):
            tpo.debug_format("Ignoring comment at line {line_num}: {line}", 6)
            continue

        # Flag ???'s as items to flesh out
        if (verbose and line.find("???") != -1):
            print "TODO: flesh out line %d: %s" % (line_num, line)

        # Check for day of week
        # TODO: make sure 'day' not included in first regex group
        ## if (my_re.search(r"^(\S+)(day)?:\s*$", line)):
        if (my_re.search(r"^(\S+)(day)?:\s*$", line)):
            tpo.debug_print("day of week check", 5)
            ## TODO: day_of_week = re.sub(r"(ur|nes)?day$", "", my_re.group(1))
            day_of_week = my_re.group(1)
            if (day_of_week in WEEKDAYS):
                if (hours > 0):
                    tpo.print_stderr("Error: missing 'hours:' line at line {n}".format(n=line_num))
            else:
                tpo.print_stderr("Error: Invalid day at line {n}".format(n=line_num))

        # Check for hours specification
        # ex: "2:30pm	4:30	debugging check_time_tracking.py"
        # Notes:
        # - The am or pm suffix is optional.
        # - If start time is greater than end time, the latter is assumed to be afternoon.
        # TODO: (my_re.search(r"^(\d+)\:?(\d*)(am|pm)?\s+(\d+)\:?(\d*)\s+\S.*", line))
        elif (my_re.search(r"^(\d+)\:?(\d*)\s+(\d+)\:?(\d*)\s+\S.*", line)):
            tpo.debug_print("time check", 5)
            # TOD: (start_hours, start_mins, start_ampm, end_hours, end_mins, start_ampm) = my_re.groups()
            (start_hours, start_mins, end_hours, end_mins) = my_re.groups()
            tpo.debug_format("sh={start_hours} sm={start_mins} eh={end_hours} em={end_mins}", 5)
            start_time = float(start_hours) + float(start_mins or 0)/60.0
            end_time = float(end_hours) + float(end_mins or 0)/60.0
            if (end_time < start_time):
                end_time += 12
            new_hours = (end_time - start_time)
            if (not (0 < new_hours <= 24)):
                tpo.print_stderr("Error: Invalid hour specification at line {n}: calculated as more than a day ({new})!".format(n=line_num, new=new_hours))
            tpo.debug_format("{new_hours} new hours", 5)
            hours += new_hours

        # Check for daily hours
        elif (my_re.search(r"^hours:\s*(\S*)", line)):
            tpo.debug_print("hours check", 5)
            specified_hours = tpo.safe_float(my_re.group(1), 0.0)
            if (specified_hours != hours):
                tpo.print_stderr("Error: Discrepancy in hours at line {n}: change {spec} => {calc}?".format(n=line_num, spec=specified_hours, calc=hours))
            weekly_hours += specified_hours
            hours = 0

        # Validate and reset weekly hours
        elif (my_re.search(r"^weekly hours:\s*(\S*)", line)):
            tpo.debug_print("weekly hours check", 5)
            specified_hours = tpo.safe_float(my_re.group(1), 0.0)
            if (specified_hours != weekly_hours):
                tpo.print_stderr("Error: Discrepancy in weekly hours at line {n}: change {spec} => {calc}?".format(n=line_num, spec=specified_hours, calc=weekly_hours))
            num_weeks += 1
            if verbose:
                print("Week %d hours: %s" % (num_weeks, weekly_hours))
            total_hours += weekly_hours
            weekly_hours = 0

        # Validate total hours
        elif (my_re.search(r"^total hours:\s*(\S*)", line)):
            tpo.debug_print("total hours check", 5)
            if (weekly_hours != 0):
                tpo.print_stderr("Error: Missing weekly hours prior to total at line {n}".format(n=line_num))
            specified_hours = tpo.safe_float(my_re.group(1), 0.0)
            if (specified_hours != total_hours):
                tpo.print_stderr("Error: Discrepancy in total hours at line {n}: change {spec} => {calc}?".format(n=line_num, spec=specified_hours, calc=total_hours))

        # Ignore miscellaneous line: starts with 5 or more dashes (e.g., "------------------...-----")
        elif (my_re.search(r"^\-\-\-\-\-+", line) or my_re.search(r"^week of", line)):
            tpo.debug_print("miscellaenous line check", 5)

        # Check for blank line without hours specification
        elif (line == ""):
            if (hours > 0):
                tpo.print_stderr("Warning: Unexpected blank line at line %d" % line_num)

        # Report lines not recognized
        else:
            tpo.print_stderr("Warning: Unexpected format at line %d: %s" % (line_num, line))

    # Do some sanoty checks
    if (hours > 0):
        tpo.print_stderr("Warning: Missing weekly total for last week")

    # Print summary
    if verbose:
        print("Total hours:  %s" % total_hours)
        weekly_average = (total_hours / num_weeks) if (num_weeks > 0) else total_hours
        print("Average per week: %s" % tpo.round_num(weekly_average))
        proto_average = (total_hours / (52 / 12.0))
        print("Proto average (i.e., 52/12): %s" % tpo.round_num(proto_average))
    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
