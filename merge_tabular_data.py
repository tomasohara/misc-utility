#! /usr/bin/env python
#
# Merge the contents of two tabular files with tab-separated values.
#
# This was originally intended to merge Wikipedia category information derived
# from the SQL file with those embedded in the wikipedia text.
#
# TODO:
# - Sample I/O
# - Rework hash reading so merging can be done at same time to avoid memory
#   overhead of second hash.
#
# Copyright (c) 2012-2018 Thomas P. O'Hara
#

"""Merge tabular data files"""

import sys
from collections import defaultdict

import debug
import system

RETAIN_CASE = system.getenv_bool("RETAIN_CASE", False)


def read_tabular_file(filename, retain_case=RETAIN_CASE):
    """Reads table with key followed by one or more tab-preceded values"""
    debug.trace_fmtd(4, "read_tabular_file({f}, [retain_case={rc}]",
                     f=filename, rc=retain_case)
    table = defaultdict(list)
    with open(filename) as f:
        for (i, line) in enumerate(f):
            if (not retain_case):
                line = line.lower().strip()
            items = line.split("\t")
            if len(items) > 1:
                table[items[0]] += items[1:]
            else:
                debug.trace_fmtd(4, "Ignoring item w/o value at line {num}",
                                 num=(i + 1))
    return table


def merge_in_hash(hash1, hash2):
    """Update HASH1 by merging in contents from HASH2; the two hashes each have list values"""
    debug.trace_fmtd(4, "merge_in_hash(_, _): len(hash1)={l1} len(hash2)={l2}",
                     l1=len(hash1), l2=len(hash2))
    for key in hash1:
        if key in hash2:
            debug.trace_fmtd(5, "Updating key {k}", k=key)
            try:
                hash1[key] += hash2[key]
            except:
                debug.trace_fmtd(4, "Error updating key '{k}' (v1={v1} v2={v2}): {exc}",
                                 k=key, v1=hash1[key], v2=hash2[key])
        else:
            debug.trace_fmtd(6, "Ignoring item '{k}' as not in second table", k=key)
    return


def merge_hashes(hash1, hash2):
    """Returns hash with merger of HASH1 and HASH2; the two hashes each have list values"""
    debug.trace_fmtd(4, "merge_hashes(_, _): len(hash1)={l1} len(hash2)={l2}",
                     l1=len(hash1), l2=len(hash2))
    new_hash = hash1.copy()
    for key in hash2:
        if key in new_hash:
            debug.trace_fmtd(5, "Updating key {k}", k=key)
            try:
                new_hash[key] += hash2[key]
            except:
                debug.trace_fmtd(4, "Error updating key '{k}' (v1={v1} v2={v2}): {exc}",
                                 k=key, v1=hash1[key], v2=hash2[key])
        else:
            debug.trace_fmtd(5, "Adding key {k}", k=key)
            new_hash[key] = hash2[key]
    debug.trace_fmtd(4, "len(new_hash)={l}",l=len(new_hash))
    return new_hash


def main(args=None):
    """Entry point for script"""
    # Check command line arguments
    if (args is None):
        args = sys.argv
    if (len(args) <= 2):
        system.print_stderr("{f}:main: two filenames for merging".
                            format(f=(__file__ or "n/a")))
        return
    filename1 = args[1]
    filename2 = args[2]

    # Read in and merge tabular data
    hash1 = read_tabular_file(filename1)
    hash2 = read_tabular_file(filename2)
    # TODO: merge_in_hash(hash1, hash2)
    combined_hash = merge_hashes(hash1, hash2)

    # Print the merged data ignoring duplicate entries
    for key in combined_hash:
        ## TODO: values = "\t".join(sorted(hash1[key]))
        try:
            values = "\t".join(combined_hash[key])
            print(key + "\t" + values)
        except:
            debug.trace_fmtd(3, "Unexpected value for combined key '{k}': '{v}",
                             k=key, v=combined_hash[key])

    return
    

#------------------------------------------------------------------------


if __name__ == '__main__':
    main()
