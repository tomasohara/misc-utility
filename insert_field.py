#! /usr/bin/env python
# 
# Inserts field into tabular data file. The values are taken from a lookup
# table created via table_lookup.py. The intention is support large tables 
# for which in-memory solutions are not feasible (e.g., via paste.perl),
# such as by using Kyoto cabinets.
#
# Usage example:
#   USE_KYOTO=1 insert_field.py --key-field-num=1 --new_-field-num=2 titles-jan-2016.ktc < 2016-01-31.rlri.spons.data > zcat 2016-01-31.rlri.spons.title.data.gz 
#

"""Adds field to tab-separated file"""

from main import Main
import tpo_common as tpo
import glue_helpers as gh
import table_lookup as tlu

UPDATE_HEADERS = "update-headers"
TABLE_FILE = "table-file"
NEW_FIELD_NAME = "new-field-name"
NEW_FIELD_NUM = "new-field-num"
KEY_FIELD_NUM = "key-field-num"
DELIM = tpo.getenv_text("DELIM", "\t", "Delimiter for input data fields")

class Script(Main):
    """Input processing class"""
    update_headers = False
    new_field_name = ""
    new_field_num = -1
    key_field_num = 1
    key_field_name = ""
    table = None
    num_fields = -1
    table_file = None
    table = None

    def setup(self):
        """Check results of command line processing"""
        tpo.debug_format("Script.setup(): self={s}", 5, s=self)
        self.update_headers = self.get_parsed_option(UPDATE_HEADERS)
        self.new_field_num = self.get_parsed_option(NEW_FIELD_NUM)
        self.new_field_name = self.get_parsed_option(NEW_FIELD_NAME)
        self.key_field_num = self.get_parsed_option(KEY_FIELD_NUM)
        self.table_file = self.get_parsed_argument(TABLE_FILE)
        gh.assertion(gh.non_empty_file(self.table_file))
        self.table = tlu.open_lookup_table(self.table_file)
        assert(self.table)
        tpo.debug_format("table size: %d", self.table.count())
        tpo.trace_object(self, label="Script instance")

    def process_line(self, line):
        """Processes current line from input"""
        fields = line.split(DELIM)
        current_num_fields = len(fields)
        value = None

        # Use first line to determine number of expected fields, etc.
        # Also, use new field name if headers given.
        if self.line_num == 1:
            self.num_fields = current_num_fields
            self.resolve_field_info()
            if self.update_headers:
                value = self.new_field_name

        # Get value for key from table file
        if value is None:
            gh.assertion(current_num_fields == self.num_fields)
            key = fields[self.key_field_num - 1]
            value = self.table.lookup([key])

        # Add new value to fields and output
        fields.insert(self.new_field_num - 1, (value if value else ""))
        print(DELIM.join(fields))

    def resolve_field_info(self):
        """Resolves position and name for new field unless specified by user"""
        assert(1 <= self.key_field_num <= self.num_fields)

        # Derive value from new field name for optional header support
        if self.update_headers and not self.new_field_name:
            # Note: empty key used to indicate label (see table_lookup.py)
            self.new_field_name = self.table.lookup([""])
            if not self.new_field_name:
                tpo.print_stderr("Warning: unable to derive header from table")
                self.new_field_name = "n/a"

        # If new field number not specified, use end
        if self.new_field_num == -1:
            self.new_field_num = 1 + self.num_fields
            tpo.debug_format("Assigning new field to end (column {c})", 3,
                             c=self.new_field_num)
        gh.assertion(1 <= self.new_field_num <= (self.num_fields + 1))


if __name__ == '__main__':
    tpo.trace_current_context(level=tpo.QUITE_DETAILED)
    app = Script(description=__doc__,
                 text_options=[(NEW_FIELD_NAME, "Name of new field to be added", Script.new_field_name)],
                 boolean_options=[(UPDATE_HEADERS, "Update first line of headers")],
                 int_options=[(NEW_FIELD_NUM, "Number of field where to place new data", Script.new_field_num),
                              (KEY_FIELD_NUM, "Number of key field from input data", Script.key_field_num)],
                 positional_arguments=[(TABLE_FILE, "File with values for new field")])
    app.run()
