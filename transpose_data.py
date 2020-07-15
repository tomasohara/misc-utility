#! /usr/bin/env python
#
# Takes an input table and transposes the rows and columns. In addition, there
# is support for showing each field value on a separate line prefixed by the
# column name. This is intended to make large table dumps more comprehensible,
# such as the database for query log information under redshift.
#------------------------------------------------------------------------
# Sample Input:
# TODO: rework samples to be non-Juju
#
#   i_timestamp   | i_ip_addr1 |             i_session_id                 | i_keyword
#   1384983367.79 | 1138872328 | 003a4a80db5eda5fa5e7359d57afc29ac1fec377 | Staples Retail Office Products
#   1384983366.04 | 1158147302 | 003b7091f121e03a4ca4e6f8b30e052f78fba19f | Quality
#   1384983366.04 | 1158147302 | 003b7091f121e03a4ca4e6f8b30e052f78fba19f | Quality
#   1384948918.84 | 1130098581 | 003bb1e9a137f6cf1ddd58941c6c7a326c9b2c3d | medical assistant
#
# Sample output:
#
#   i_timestamp: 1384983367.79 | 1384983366.04 | 1384983366.04 | 1384948918.84
#   i_ip_addr1: 1138872328 | 1158147302 | . | 1130098581
#   i_session_id: 003a4a80db5eda5fa5e7359d57afc29ac1fec377 | 003b7091f121e03a4ca4e6f8b30e052f78fba19f | . | 003bb1e9a137f6cf1ddd58941c6c7a326c9b2c3d
#   i_keyword: Staples Retail Office Products | Quality | . | medical assistant
#
#   via: ./transpose-data.py --elide --delim=' | ' < sample-transpose-input.data 
#------------------------------------------------------------------------
# Illustration of using script to extract top sessions from query logs (via Bash)
# 
# - Extract data from redshift and format in tab-delimited formtat
#   note: based on first 100,000 from ____qiqcix table under redshift
#
#    $ /usr/bin/time echo 'SELECT * FROM ____qiqcix LIMIT 100000;' | psql -h rsjuju.cjhlfgrlo68q.us-east-1.redshift.amazonaws.com -p 5439 rsjuju >| redshift-____qiqcix.first-100K.data 2>| redshift-____qiqcix.first-100K.data.log
#
# - Extract information for top-40 sessions by request count, saving each in separate trasposed file.
#    $ export TOPN=40; (
#      logfile=redshift-____qiqcix.first-100K.data
#      outdir=/tmp/top$TOPN-session
#      mkdir $outdir
#      head -1 $logfile > $outdir/$logfile.header
#      cut -f9 -d'|' $logfile | count_it.perl '\S+' - > $outdir/session_id.freq
#      head -$TOPN session_id.freq | cut -f1 > $outdir/session_id.list
#      for session in `cat $outdir/session_id.list`; do 
#         grep $session $logfile > $outdir/$session.data
#         transpose-data.py --elide --delim=' | ' --header=$logfile.header < $outdir/$session.data > $outdir/$session.transpose.data
#      done
#    ) > extract-$TOPN-sessions.log 2>&1
#    
#------------------------------------------------------------------------
# Notes:
# - This was developed to support the analysis done for Ticket #13777 (robot-resilient
# report for a/b hypothesis tests): see https://trac.juju.com/ticket/13777.
# - See the ticket for extraction illustration that accounts for impression vs click session id.
#
#------------------------------------------------------------------------
# TODO:
# - Have option to disable use of labels alltogether.
# - Have option to prefix values with column number.
# - Ignore (first) header line if same as headers option.
# - Have separate option for output delim (eg., ": ").
# - Rework field extraction using csv package (rather than split).
# - Have option for setting delim to tab to avoid awkward spec under bash (e.g., --delim $'\t').
#

"""Takes an input table and transposes the rows and columns"""

import sys
import argparse

from tpo_common import debug_print, trace_array, print_stderr
from glue_helpers import read_lines

def main():
    """Entry point for script"""
    # Check command-line arguments
    parser = argparse.ArgumentParser(description="Transpose a table from input")
    parser.add_argument("--header", help="File with field names")
    parser.add_argument("--delim", help="Delimiter for fields")
    parser.add_argument("--elide", dest='elide_fields', action='store_true', default=False, help="Replace repeated values by .'s")
    parser.add_argument("--elided-value", help="Value for repeated field")
    parser.add_argument("--single-field", dest='single_field', action='store_true', default=False, help="Only show a single field per output line")
    parser.add_argument("filename", nargs='?', default='-')
    args = vars(parser.parse_args())
    debug_print("args = %s" % args, 5)
    delim = "\t"
    elided_value = "."
    field_names = []
    field_data = []
    single_field = args['single_field']
    elide_fields = args['elide_fields']
    if args['delim']:
        delim = args['delim']
    if args['elided_value']:
        elided_value = args['elided_value']
    if args['header']:
        lines = read_lines(args['header'])
        field_names = [label.strip() for label in lines[0].split(delim)]
        trace_array(field_names, 5, "field_names")
        if not single_field:
            field_data = [[] for i in range(len(field_names))]
            trace_array(field_data, 5, "field_data")
        previous_value = [None] * len(field_names)
    input_stream = sys.stdin
    if (args['filename'] and (args['filename'] != "-")):
        input_stream = open(args['filename'])

    # Transpose each line of the table
    num_lines = 0
    for line in input_stream:
        num_lines += 1
        line = line.strip("\n")
        debug_print("L%d: %s" % (num_lines, line), 6)
        line_data = [field.strip() for field in line.split(delim)]
        trace_array(line_data, 5, "line_data")

        # Use first line as field names if not yet defined
        if (len(field_names) == 0):
            field_names = line_data
            if not single_field:
                field_data = [[] for i in range(len(field_names))]
                trace_array(field_data, 5, "field_data")
            previous_value = [None] * len(field_names)
            continue
        elif ((num_lines == 1) and (field_names == line_data)):
            debug_print("Ignoring duplicate header", 5)
            continue

        # Append each field to respective list (of seen values)
        if (len(line_data) != len(field_names)):
            print_stderr("Warning: Found %d fields but expected %d" % (len(line_data), len(field_names)))
            line_data += (['n/a'] * max(0, len(field_names) - len(line_data)))
        for i in range(len(field_names)):
            debug_print("d[%d]: %s" % (i, line_data[i]), 7)
            new_value = line_data[i]
            if (elide_fields and (previous_value[i] == line_data[i])):
                new_value = elided_value
            if (single_field):
                print("%s" % (delim.join([field_names[i], new_value])))
            else:
                field_data[i].append(new_value)
                previous_value[i] = line_data[i]
        if not single_field:
            trace_array(field_data, 8, "field_data")

    # Output the transposed lines
    if not single_field:
        for i in range(len(field_names)):
            print("%s" % delim.join([field_names[i]] + field_data[i]))

    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
