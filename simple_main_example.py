#! /usr/bin/env python
# 
# Sample script using Main class.
#

"""Simple illustration of Main class"""

import re
from main import Main
import tpo_common as tpo
import glue_helpers as gh

class Script(Main):
    """Input processing class"""
    regex = None
    check_fubar = None

    def setup(self):
        """Check results of command line processing"""
        self.regex = self.parsed_args['regex']
        self.check_fubar = self.get_parsed_option('check-fubar', not self.regex)
        gh.assertion(bool(self.regex) ^ self.check_fubar)
        tpo.trace_object(self, label="Script instance")

    def process_line(self, line):
        """Processes current line from input"""
        if self.check_fubar and "fubar" in line:
            tpo.debug_format("Fubar line {n}: {l}", 4, n=self.line_num, l=line)
            print(line)
        elif self.regex and re.search(self.regex, line):
            tpo.debug_format("Regex line {n}: {l}", 4, n=self.line_num, l=line)
            print(line)
        return

if __name__ == '__main__':
    app = Script(description=__doc__,
                 boolean_options=["check-fubar"],
                 text_options=[("regex", "Regular expression")])
    app.run()
