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
# 2	3	appserver overview by Andy; diagnosing hg update issues on systen repository
# 5	6	re-cloning and reviewing system repository source structure
# 7	8	adding new repository branch; checking old system repository differences (e.g., TODO notes in code)
# 9:30	11:30	exporting related-title code from search prototype into appserver; adding prequisites for running appserver locally (for syntax checking)
# 12	2	trying to get @route-based dispatching for relatedJobs method
# hours: 7
#
#-------------------------------------------------------------------------------
# TODO:
# - ** Add edit distance check for 'break' mispellings (e.g., breal) to avoid recording excess time!!
# - Allow _ placeholders in week and total summary if month not complete.
# - Clean up debug_format calls.
# - Make hours line optional (e.g., assume blank line separates days).
# - Make weekly hours optional (e.g., assume repeated day separates week).
# - Have option to output template for current month!
#
#

"""Validates the hour tabulations in a time tracking report"""

import argparse
import os
import re
import sys

import debug
## OLD: import tpo_common as tpo
import sys_version_info_hack                  # pylint: disable=unused-import
import system
from my_regex import my_re

# Note: python 3.6+ format strings are used (n.b., assumes sys_version_info_hack above)
# TODO: See if why to have pylist issue warning about version; that way, there's less
# head scratching about syntax error messages (e.g., for python-3.6+ f-string interpolation).
assert((sys.version_info.major >= 3) and (sys.version_info.minor >= 6))

WEEKDAYS = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
ABBREVIATED_WEEKDAYS = ["sun", "mon", "tues", "wed", "thurs", "fri", "sat"]
ALL_WEEKDAYS = WEEKDAYS + ABBREVIATED_WEEKDAYS

#................................................................................

def main():
    """Entry point for script"""
    debug.trace(4, f"main(): argv={sys.argv}")

    # Check command-line arguments
    # TODO: Switch over to argument processing via Main class in main.py.
    # TODO: Add example illustrating DEBUG_LEVEL and to show check-time-tracking alias;
    # also, add option to output blank template for the month.
    parser = argparse.ArgumentParser(description="Validates time tracking reports")
    parser.add_argument("--strict", default=False, action='store_true', help="strict parsing of the time tracking report")
    parser.add_argument("--ignore-comments", default=False, action='store_true', help="ignore comments in the time tracking report")
    parser.add_argument("--verbose", default=True, action='store_true', help="verbose output mode (e.g., summary of weekly hours)")
    parser.add_argument("--quiet", dest='verbose', action='store_false', help="non-verbose mode")
    parser.add_argument("--weekly", dest='weekly', default=False, action='store_true', help="show weekly summary of hours")
    parser.add_argument("--heuristics", default=False, dest='heuristics', action='store_true', help="use heuristics such as ignoring time slots labelled as 'break' (n.b., can be confusing without --quiet or debugging enabled)")
    parser.add_argument("--skip-hours-check", default=False, action='store_true', help="TODO: does xyz, ...")
    ## TODO: parser.add_argument("--xyz", default=False, dest='xyz', action='store_true', help="TODO: does xyz, ...")
    ## -or- parser.add_argument("--not-xyz", dest='xyz', action='store_false', help="TODO: does not xyz, ...")
    parser.add_argument("filename", help="input filename")
    args = vars(parser.parse_args())
    debug.trace(5, f"args = {args}")
    filename = args['filename']
    debug.assertion(os.path.exists(filename))
    strict = args['strict']
    ignore_comments = args['ignore_comments'] or (not strict)
    use_heuristics = args['heuristics']
    verbose = args['verbose']
    show_weekly = args['weekly']
    skip_hours_check = args['skip_hours_check']
    ## TODO: verbose = args['verbose'] or use_heuristics

    # Print header for weekly summary and initialize associated record keeping
    if show_weekly:
        print("\t".join(ABBREVIATED_WEEKDAYS))
    weekday_hours = {}
    day_of_week = ""
    
    # Scan time listing keeping track of hours running totals
    hours = 0
    weekly_hours = 0
    total_hours = 0
    line_num = 0
    num_weeks = 0
    for line in open(filename):
        line = line.strip("\n")
        line_num += 1
        debug.trace_fmt(6, f"L{line_num}: {line}")

        # Normalize line to facilitate extraction:
        # - treat whitespace delimiters as tabs after start and end time
        # - remove surrounding whitespace, make lowercase
        # - remove am/pm suffixes in time specifications
        # TODO: convert "T1-T2<tab>..." to "T1<tab>T2<tab>..."
        line = line.lower().strip()
        if (re.search(r"^(\d+\S*)\s+(\d+\S*)\s+", line)):
            line = re.sub(r"^(\S+)\s+(\S+)\s+", r"\1\t\2\t", line)
            debug.assertion(not line.endswith("\t"))
            debug.assertion(not re.search(r"\t.*\t.*\t", line))
        line = re.sub(r"(am|pm)\t", r"\t", line)
        debug.trace_fmt(6, f"Mid-normalized L{line_num}: {line}")

        # Ignore comments (line starting either with '#' or ';' ignoring leading space)
        if (ignore_comments and re.search(r"^\s*[#;]", line)):
            debug.trace_fmt(6, f"Ignoring comment at line {line_num}: {line}")
            continue

        # Strip trailing comments
        # ex: "Fri:	    # 28" => "Fri:"
        line = re.sub(r"\s*[#;].*", "", line)
        debug.trace_fmt(5, f"Final normalized L{line_num}: {line}")

        # Flag ???'s as items to flesh out
        if (verbose and line.find("???") != -1):
            print("TODO: flesh out line %d: %s" % (line_num, line))

        # Apply heuristics for entries to ignore, etc. (e.g., ignore time-slots listed as break's).
        if use_heuristics and re.search(r"\tbreak\s*$", line):
            message = "Ignoring break at line {n}: {t}".format(n=line_num, t=line)
            if verbose:
                print(message)
            debug.trace(4, f"Warning: {message}")
            continue
            
        # TEMP HACK: first Check for daily hours (redundant with old code below)
        # TODO: See why not handled properly in time-tracking-mmmYY.template
        # ex: "hours: 8.5"
        if (my_re.search(r"^hours:\s*(\S*)", line)):
            debug.trace(5, "hours check")
            specified_hours = system.safe_float(my_re.group(1), 0.0)
            ## OLD: if (specified_hours != hours):
            if ((specified_hours != hours) and (not skip_hours_check)):
                system.print_stderr("Error: Discrepancy in hours at line {n}: change {spec} => {calc}?".format(n=line_num, spec=specified_hours, calc=hours))
            else:
                # HACK: pretend user specified tabulated hours (TODO: use 'hours' below for clarify)
                specified_hours = hours
            weekly_hours += specified_hours
            weekday_hours[day_of_week] = specified_hours
            hours = 0

        # Check for day of week
        # TODO: make sure 'day' not included in first regex group
        ## if (my_re.search(r"^(\S+)(day)?:\s*$", line)):
        ## OLD: if (my_re.search(r"^(\S+)(day)?:\s*$", line)):
        elif (my_re.search(r"^(\S+)(day)?:\s*$", line)):
            day_of_week = my_re.group(1)
            debug.trace_fmt(5, f"day of week check: day={day_of_week}")
            ## TODO: handle Saturday and Wednesday
            ## TODO: day_of_week = re.sub(r"(ur|nes)?day$", "", my_re.group(1))
            if (day_of_week in ALL_WEEKDAYS):
                if (hours > 0):
                    system.print_stderr("Error: missing 'hours:' line at line {n}".format(n=line_num))
            else:
                system.print_stderr("Error: Invalid day '{d}' at line {n}", d=day_of_week, n=line_num)

        # Check for hours specification
        # ex: "2:30pm	4:30	debugging check_time_tracking.py"
        # ex: "7-9	trying to get SpaCy NER via Anaconda under edgenode and win10
        # Notes:
        # - The am or pm suffix is optional.
        # - If start time is greater than end time, the latter is assumed to be afternoon.
        # - The start and end time can optionally be separate by a dash (e.g, 8-10pm), without intervening spaces (e.g., not 8 - 10pm)
        #
        # TODO: (my_re.search(r"^(\d+)\:?(\d*)(am|pm)?\s+(\d+)\:?(\d*)\s+\S.*", line))
        elif (my_re.search(r"^(\d+)\:?(\d*)\s+(\d+)\:?(\d*)\s+\S.*", line) or
              my_re.search(r"^(\d+)\:?(\d*)\-(\d+)\:?(\d*)\s+\S.*", line)):
            debug.trace(5, "time check")
            # TOD: (start_hours, start_mins, start_ampm, end_hours, end_mins, start_ampm) = my_re.groups()
            (start_hours, start_mins, end_hours, end_mins) = my_re.groups()
            debug.trace_fmt(5, f"sh={start_hours} sm={start_mins} eh={end_hours} em={end_mins}")
            start_time = float(start_hours) + float(start_mins or 0)/60.0
            end_time = float(end_hours) + float(end_mins or 0)/60.0
            if (end_time < start_time):
                end_time += 12
            new_hours = (end_time - start_time)
            if (not (0 < new_hours <= 24)):
                system.print_stderr("Error: Invalid hour specification at line {n}: calculated as more than a day ({new})!".format(n=line_num, new=new_hours))
            debug.trace_fmt(5, f"{new_hours} new hours")
            hours += new_hours


        # Check for daily hours
        # HACK: keep in synch with (temp) hack above
        elif (my_re.search(r"^hours:\s*(\S*)", line)):
            debug.trace(5, "hours check")
            specified_hours = system.safe_float(my_re.group(1), 0.0)
            ## OLD: if (specified_hours != hours):
            if ((specified_hours != hours) and (not skip_hours_check)):
                system.print_stderr("Error: Discrepancy in hours at line {n}: change {spec} => {calc}?".format(n=line_num, spec=specified_hours, calc=hours))
            else:
                # HACK: pretend user specified tabulated hours (TODO: use 'hours' below for clarify)
                specified_hours = hours
            weekly_hours += specified_hours
            weekday_hours[day_of_week] = specified_hours
            hours = 0

        # Validate and reset weekly hours
        elif (my_re.search(r"^weekly hours:\s*(\S*)", line)):
            debug.trace(5, "weekly hours check")
            specified_hours = system.safe_float(my_re.group(1), 0.0)
            ## OLD: if (specified_hours != weekly_hours):
            if ((specified_hours != weekly_hours) and (not skip_hours_check)):
                system.print_stderr("Error: Discrepancy in weekly hours at line {n}: change {spec} => {calc}?".format(n=line_num, spec=specified_hours, calc=weekly_hours))
            num_weeks += 1
            if verbose:
                print("Week %d hours: %s" % (num_weeks, weekly_hours))
            total_hours += weekly_hours
            weekly_hours = 0

            # Show hours by day of week and then reset associated record keeping
            # TODO: use map in place of simple list comprehensions
            if show_weekly:
                hour_per_day = [weekday_hours.get(d, 0) for d in ABBREVIATED_WEEKDAYS]
                print("\t".join([system.to_string(h) for h in hour_per_day]))
                weekday_hours = {}
                day_of_week = ""

        # Validate total hours
        elif (my_re.search(r"^total hours:\s*(\S*)", line)):
            debug.trace(5, "total hours check")
            if (weekly_hours != 0):
                system.print_stderr("Error: Missing weekly hours prior to total at line {n}".format(n=line_num))
            specified_hours = system.safe_float(my_re.group(1), 0.0)
            ## OLD: if (specified_hours != total_hours):
            if ((specified_hours != total_hours) and (not skip_hours_check)):
                system.print_stderr("Error: Discrepancy in total hours at line {n}: change {spec} => {calc}?".format(n=line_num, spec=specified_hours, calc=total_hours))

        # Ignore miscellaneous line: starts with 5 or more dashes (e.g., "------------------...-----")
        elif (my_re.search(r"^\-\-\-\-\-+", line) or my_re.search(r"^week of", line)):
            debug.trace(5, "miscellaenous line check")

        # Check for blank line without hours specification
        elif (line == ""):
            debug.trace(5, "blank line check")
            if (hours > 0):
                system.print_stderr("Warning: Unexpected blank line at line %d" % line_num)

        # Report lines not recognized
        else:
            system.print_stderr("Warning: Unexpected format at line %d: %s" % (line_num, line))

    # Do some sanoty checks
    if (hours > 0):
        system.print_stderr("Warning: Missing weekly total for last week")

    # Print summary
    if verbose:
        print("Total hours:  %s" % total_hours)
        weekly_average = (total_hours / num_weeks) if (num_weeks > 0) else total_hours
        print("Average per week: %s" % system.round_num(weekly_average))
        proto_average = (total_hours / (52 / 12.0))
        print("Proto average (i.e., 52/12): %s" % system.round_num(proto_average))
    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
