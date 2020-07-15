#! /usr/bin/env python
#
# Extracts columns from input file into separate output files. The
# filename is optionally given in the first column, and the file data
# is based on the list of fields (e.g., 2+). (TODO, if no filename
# column is supplied, the names will be file-N.EXT (e.g., file-1.xml).)
#

"""Extract columns into separate files (e.g., first column for filename and other columns for contents)"""

import re
import pandas as pd

from main import Main
import debug
import system
import tpo_common as tpo

DELIM = system.getenv_text("DELIM", ",") 

FILENAME_COLUMN = "filename-column"
DATA_COLUMNS = "data-columns"


def get_class_name(instance):
    """Get name of class for INSTANCE"""
    ## BAD: name = (getattr(instance, "__class__", "???").replace("__main__.", ""))
    name = (str(getattr(instance, "__class__", "???")).replace("__main__.", ""))
    debug.trace_fmt(5, "get_class_name({inst}) => {n}", inst=instance, n=name)
    return name


def has_no_spaces(text):
    """Indicates whether TEXT has any whitespace"""
    # EX: (not has_no_spaces("\r"))
    no_spaces = (not re.search(r"\s", text))
    debug.trace_fmt(5, "has_no_spaces({t}) => {r}", t=text, r=no_spaces)
    return no_spaces


class ExtractSubfiles(Main):
    """Class for extracting subfile for each line in CSV input"""
    filename_column = 0
    data_column_spec = "-1"
    data_columns = []

    def setup(self):
        """Process arguments: get column number for filename and column numbers for data"""
        debug.trace_fmt(5, "ExtractSubfiles.setup(); self={s}", s=self)
        self.filename_column = tpo.safe_int(self.get_parsed_option(FILENAME_COLUMN, self.filename_column))
        data_column_spec = self.get_parsed_option(DATA_COLUMNS, self.data_column_spec).replace(",", " ")
        self.data_columns = [tpo.safe_int(v) for v in data_column_spec.split()]
        ## OLD: debug.assertion(all([isinstance(v, int) for v in self.data_columns]))
        debug.assertion(self.filename_column not in self.data_columns)
        # TODO: define get_class_name helper
        debug.trace_object(6, self, "{cl} instance".format(cl=get_class_name(self)))
        return

    def run_main_step(self):
        """Main processing: reads file and outputs each line to a separate file"""
        debug.trace_fmt(5, "ExtractSubfiles.run_main_step(); self={s}", s=self)
        processed_files = set()
        df = pd.read_csv(self.filename, sep=DELIM, dtype=str)
        debug.trace_fmt(5, "type(df)={t}", t=type(df))

        # Determine the column labels for the filename and for the data
        # Note: column labels shouldn't have whitespace (TODO, just prohibit tab)
        labels = list(df)
        debug.trace_fmt(5, "type(labels)={t}", t=type(labels))
        debug.assertion(all([has_no_spaces(l) for l in labels]))
        filename_label = labels[self.filename_column]
        use_all_other_labels = (self.data_columns == [-1])
        other_labels = []
        for i, label in enumerate(labels):
            if (label == filename_label):
                continue
            if (use_all_other_labels or (i in self.data_columns)):
                other_labels.append(label)
        debug.assertion(filename_label not in other_labels)

        # Output each line as a separate file
        for r in range(len(df)):
            try:
                row = df.iloc(r)
            except(ValueError):
                break
            debug.trace_fmt(5, "type(row)={t}", t=type(row))
            ## BAD: filename = row[0]
            filename = row[0][0]
            ## BAD: debug.assertion(all([has_no_spaces(v) for v in row[:1]]))
            ## BAD: data = " ".join([str(v) for v in row[:1]])
            data = ""
            for i in range(len(row[0]) - 1):
                if (i > 0):
                    data += " "
                data += str(row[0][i + 1])
            debug.trace_fmt(7, "Writing file '{f}' with data: {d}", f=filename, d=data)
            system.write_file(filename, data)
            debug.assertion(filename not in processed_files)
            processed_files.add(filename)
        return

if __name__ == '__main__':
    app = ExtractSubfiles(description=__doc__,
                          skip_input=True,
                          manual_input=True,
                          text_options=[(FILENAME_COLUMN, "Column number to use for filename (e.g., 1)"),
                                        (DATA_COLUMNS, "Column numbers to retain (-1 for all except filename column)")])
    app.run()
