#! /usr/bin/env python
# 
# Filters lines in input file, based on random numbers
# TODO: use iterator for input (see 
#

"""Filter lines randomly"""

import random
from main import Main
import tpo_common as tpo

INCLUDE_HEADER = "include-header"
RATIO = "ratio"
SEED = "seed"
DEFAULT_RATIO = tpo.getenv_number("DEFAULT_RATIO", 0.10)
RANDOM_SEED = tpo.getenv_integer("RANDOM_SEED", None, 
                                 "Integral seed for randoom number generation")

class Filter(Main):
    """Input processing class"""
    include_header = False
    ratio = 0.10

    def setup(self):
        """Process arguments"""
        self.include_header = self.get_parsed_option(INCLUDE_HEADER)
        self.ratio = self.get_parsed_option(RATIO)
        seed = self.get_parsed_option(SEED)
        if seed:
            random.seed(seed)
        tpo.trace_object(self, 6, "filter instance")
        tpo.debug_format("ratio={r}, seed={s}", 4, 
                         r=self.ratio, s=seed)

    def process_line(self, line):
        """Processes current line from input (randomly printing)"""
        include = ((random.random() <= self.ratio)
                   or ((self.line_num == 1) and (self.include_header)))
        tpo.debug_format("include={incl}", 6, incl=include)
        if include:
            print(line)
        return

if __name__ == '__main__':
    app = Filter(description=__doc__,
                 # TODO: use Main.read_input directly w/ manual_input=True
                 boolean_options=[(INCLUDE_HEADER, "Include header line")],
                 float_options=[(RATIO, "Random threshoold", DEFAULT_RATIO), 
                                (SEED, "Random seed", RANDOM_SEED)])
    app.run()
