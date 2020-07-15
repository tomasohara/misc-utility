#! /usr/bin/env python
#
# Converts the values in a table, so that cells are relative to a reference
# cell.
#
# Sample input:
#   cat 1	10
#   cat 2	20
#
# Sample output:
#   cat 1	10 (*)
#   cat 2	20 (1.0)
#
# Note:
# - This is based on code from the A/B test based on query log analysis:
#   see postproc_ab.py in src/query-log-analysis/get_logistic_regression_inputs.
#

import sys
import re
import argparse

from tpo_common import debug_print, debug_format

def round_num(num, precision=3):
    """Rounds NUM to PRECISION places (default of 3)"""
    return (round(num, precision))

def main():
    """Entry point for script"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Converts valus in table to be relative to reference row(s)")
    parser.add_argument("--label-field", type=int, default=1, help="Column number for row label")
    parser.add_argument("--cat-regex", default="\S+", help="Pattern for deriving category from label field ")
    parser.add_argument("--delim", default="\t", help="Delimiter for fields")
    parser.add_argument("--ignore-cats", default="", help="String list of categories to ignore")
    parser.add_argument("--ref-cat", default=None, help="Reference category for deriving relative differences")
    parser.add_argument("--header", default=True, action='store_true', help="Table includes header row (default)")
    parser.add_argument("--no-header", dest='header', default=False, action='store_false', help="Table doesn't have header row")
    parser.add_argument("--just-diff", default=False, action='store_true', help="Only displays the relative differences")
    
    parser.add_argument("filename", nargs='?', default='-', help="Input filename")
    args = vars(parser.parse_args())
    debug_print("args = %s" % args, 5)
    ## TODO: delim = args.delim
    delim = args['delim'] if 'delim' in args else "\t"
    input_stream = sys.stdin
    if (args['filename'] and (args['filename'] != "-")):
        input_stream = open(args['filename'])
    label_field_num = args['label_field']
    cat_regex = args['cat_regex']
    has_header = args['header']
    ignore_cats = re.split("\s+", args['ignore_cats'])
    ref_cat = args['ref_cat']
    just_difference = args['just_diff']
    
    # Make sure regex includes grouping parentheses
    debug_print("cat_regex: %s" % cat_regex, 5)
    regex = re.compile(cat_regex)
    if (regex.groups == 0):
        regex = re.compile("(" + cat_regex + ")")

    # Read the table, keeping track of the row categories.
    # Note: in postproc_ab.py the algorithm part of the tag would be the category.
    rows = []
    row_cats = []
    label_row_offset = {}
    num_lines = 0
    for line in input_stream:
        line = line.strip("\n")
        num_lines += 1
        debug_print("L%d: %s" % (num_lines, line), 6)

        # Convert line to row and make sure each cell is numeric except for labels
        if (not line):
            debug_print("Ignoring blank line (line %d)" % num_lines, 4)
            continue
        data = line.split("\t")
        assert((len(rows) == 0) or (len(data) == len(rows[-1])))
        if (has_header and (len(rows) == 0)):
            rows.append(data)
            row_cats.append(None)
            continue
        for c in range(len(data)):
            data[c] = float(data[c]) if (c != (label_field_num - 1)) else data[c]

        # Derive category for row (e.g., via row label or suffix of it)
        label = data[label_field_num - 1]
        label_row_offset[label] = len(rows)
        match = regex.search(label)
        if match:
            row_cat = match.group(1)
        else:
            debug_print("Warning: Unable to derive category for row %d: row_cat=%s" % (1 + len(rows), label))
            row_cat = label
        debug_print("cat: %s; data: %s" % (row_cat, data), 5)

        # Skip row if in categories to ignore
        if (row_cat in ignore_cats):
            debug_format("Ignoring row for category {row_cat} at line {num_lines}", 4)
            continue

        # Make note of first row category encountered as reference category
        if not ref_cat:
            ref_cat = row_cat
            debug_print("Reference category: %s" % ref_cat, 3)

        # Update table
        rows.append(data)
        row_cats.append(row_cat)
    debug_print("label_row_offset=%s" % label_row_offset, 5)

    # Convert the values from absolute to relative
    rel_rows = []
    for r in range(len(rows)):
        row = list(rows[r])
        rel_row = row
        if (has_header and (r == 0)):
            rel_rows.append(rel_row)
            continue
        label = row[label_field_num - 1]
        row_cat = row_cats[r]
        debug_print("row[{r}] = {row}; label={label}; cat={row_cat}".format(**locals()), 4)

        # Compute relative difference valus
        if row_cat == ref_cat:
            diffs = ["*"] * len(row)
        else:
            # Determine reference row by substituting reference cat for current cat in row label.
            # Get corresponding values, accounting for possibly incomplete table.
            ref_label = label.replace(row_cat, ref_cat)
            assert(ref_label != label)
            if ref_label not in label_row_offset:
                debug_print("Warning: no data for reference label %s" % ref_label)
                ref_row = [0] * len(row)
                # Note: Adds a dummy relative row for missing reference row
                ref_row[label_field_num - 1] = "[" + ref_label + "]"
                rel_rows.append(ref_row)
            else:
                assert(label_row_offset[ref_label] != r)
                ref_row = rows[label_row_offset[ref_label]]
                debug_print("ref(%d): %s" % (label_row_offset[ref_label], ref_row), 4)
            # Compute diff's for each cell, ignoring label
            diffs = [""]
            for c in range(len(row)):
                if (c != (label_field_num - 1)):
                    rel_diff = round_num((row[c] - ref_row[c])/float(ref_row[c])) if ref_row[c] else "n/a"
                    diffs.append(rel_diff)
        debug_print("diffs: %s" % diffs, 4)

        # Update row values
        for c in range(len(row)):
            if (c != (label_field_num - 1)):
                if just_difference:
                    rel_row[c] = diffs[c]
                else:
                    rel_row[c] = "%s (%s)" % (row[c], diffs[c])
        rel_rows.append(rel_row)

    # Output the revised table
    for row in rel_rows:
        print("\t".join([str(v) for v in row]))

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
