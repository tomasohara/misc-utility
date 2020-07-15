#! /usr/bin/env python
# 
# Similar to Unix cut command but with support for CSV files. Also modelled
# after perl script with support for treating runs of whitespace as tab.
#
# Notes:
# - Input processing based on csv example (seee https://docs.python.org/2/library/csv.html).
# - The bulk of the work is parsing the field specification After that,
#   the processing is simple row readidng and column extraction (i.e., subsetting).
# - Have option for setting delim to tab to avoid awkward spec under bash (e.g., --output-delim $'\t').

#
# TODO:
# - Add option for converting the field separator (e.g., input csv and output tab-separated).
#

"""Extracts columns from a file as with Unix cut command"""

import csv
import re
import sys

from main import Main
from regex import my_re
import debug
import system

# Fill out constants for switches omitting leading dashes (e.g., DEBUG_MODE = "debug-mode")
FIELDS = "fields"
## TODO: F = "f"                        # alias for --fields
FIX = "fix"                             # convert runs of spaces into a tab
CSV = "csv"                             # comma-separated value format
OUT_DELIM = "output-delim"              # output delimiter if not same for input
ALL_FIELDS = "all-fields"               # use all fields in output (e.g., for deimiter conversion)
TAB = "\t"

class Script(Main):
    """Input processing class"""
    fields = []
    fix = False
    delimiter = TAB
    output_delimiter = delimiter
    csv = False
    csv_reader = None
    all_fields = False

    def setup(self):
        """Check results of command line processing"""
        debug.trace_fmtd(5, "Script.setup(): self={s}", s=self)
        fields = self.get_parsed_option(FIELDS, self.fields)
        if fields:
            self.fields = self.parse_field_spec(fields)
        self.all_fields = self.get_parsed_option(ALL_FIELDS, (not fields))
        debug.assertion(not (self.fields and self.all_fields))
        self.fix = self.get_parsed_option(FIX, self.fix)
        self.csv = self.get_parsed_option(CSV, self.csv)
        self.delimiter = "\t"
        if self.csv:
            self.delimiter = ","
        self.output_delimiter = self.get_parsed_option(OUT_DELIM, self.output_delimiter)
        # TODO: use iter(sys.stdin.read())??
        ## BAD: self.csv_reader = csv.reader(iter(system.stdin_reader()), delimiter=self.delimiter, quotechar='"')
        self.csv_reader = csv.reader(iter(sys.stdin.readlines()), delimiter=self.delimiter, quotechar='"')
        # TODO: see if there is an option to determine number of fields (before reading data)
        ## if self.all_fields:
        ##    self.fields = range(self.csv_reader.num_fields)
        debug.trace_object(5, self, label="Script instance")
        return

    def parse_field_spec(self, field_spec):
        """Convert the FIELD_SPEC from string to list of integers. The specification can contain numeric ranges (e.g., "3-5") or comma-separated values (e.g., "7,9,11").
        Note: throws exception if fields are not integers"""
        # Replace ranges with comma-separated cols (e.g., "5-7" => "5,6,7")
        debug.trace_fmtd(5, "parse_field_spec({fs})", fs=field_spec)
        if "-" in field_spec:
            while my_re.search(r"([0-9]+)\-([0-9]+)", field_spec):
                range_spec = my_re.group(0)
                debug.trace_fmtd(4, "Converting range: {r}", r=range_spec)
                start = int(my_re.group(1))
                end = int(my_re.group(2))
                # Add comma-separated values in range
                subfield_spec = ""
                while (start < end):
                    subfield_spec += start + ","
                    start += 1
                if subfield_spec.endswith(self.delimiter):
                    subfield_spec = subfield_spec[:-1]
                # Replace the range spec. with the delimited values
                field_spec = field_spec.replace(range_spec, subfield_spec, 1)
        # Convert text field specification into list
        field_list = []
        if field_spec:
            field_list = [int(f) for f in re.split(r"[, ]", field_spec)]
        debug.assertion(field_list)
        debug.trace_fmtd(4, "parse_field_spec() => {fl}", fl=field_list)
        return field_list
    
    def run_main_step(self):
        """Main processing step: read each line (i.e. row) and extract specified columns.
        Note: The fields are 1-based (i.e., first column specified 1 not 0)"""
        debug.trace_fmtd(4, "run_main_step()")
        for i, row in enumerate(self.csv_reader):
            debug.trace_fmt(6, "R{n}: {r}", n=(i + 1), r=row)
            # Derive the fields to extract if all to be extracted
            debug.trace_fmt(7, "pre f={f} all={a}", f=self.fields, a=self.all_fields)
            if ((not self.fields) and self.all_fields):
                self.fields = [(c + 1) for c in range(len(row))]
                if not self.fields:
                    system.print_stderr("Error: No items in header row at line {l}", l=(i + 1))
            debug.trace_fmt(7, "post f={f}", f=self.fields)
            debug.assertion(self.fields)

            # Output line with fields joined by (output) separator
            line = ""
            for f in self.fields:
                line += (row[f - 1] if (1 <= f <= len(row)) else "")
                line += self.output_delimiter
            if line.endswith(self.output_delimiter):
                line = line[:-1]
            print(line)
        return

if __name__ == '__main__':
    debug.trace_current_context()
    app = Script(
        description=__doc__,
        skip_input=True,
        boolean_options=[CSV, FIX, ALL_FIELDS],
        text_options=[(OUT_DELIM, "Output field separator"),
                      (FIELDS, "Field specification (1-based): single column, range of columns, or comma-separated columns")])
    app.run()
