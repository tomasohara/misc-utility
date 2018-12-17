# Convenience class for regex searching, providing simple wrapper around
# static match results.
#
# Example usage:
#
#    from regex import my_re
#    ...
#    if (my_re.search(r"^(\d+)\:?(\d*)\s+(\d+)\:?(\d*)\s+\S.*", line)):
#        (start_hours, start_mins, end_hours, end_mins) = my_re.groups()
#
#
# Copyright (c) 2012-2018 Thomas P. O'Hara
#

"""Wrapper class for regex match results"""

import re

import debug
import tpo_common as tpo

class regex(object):
    """Class implementing regex search that saves match results
    note: Allows regex to be used directly in conditions"""
    # TODO: IGNORECASE = re.IGNORECASE, etc.

    def __init__(self, ):
        tpo.debug_format("regex.__init__(): self={s}", 4, s=self)
        self.match_result = None
        # TODO: self.regex = ""

    def search(self, regex, text, flags=0):
        """Search for REGEX in TEXT with optional FLAGS"""
        tpo.debug_format("regex.search({r}, {t}, {f}): self={s}", 7, 
                         r=regex, t=text, f=flags, s=self)
        self.match_result = re.search(regex, text, flags)
        if self.match_result:
            tpo.debug_print("match: %s" % tpo.to_string(self.match_result.groups()), 6)
        return self.match_result

    def match(self, regex, text, flags=0):
        """Match REGEX to TEXT with optional FLAGS"""
        tpo.debug_format("regex.match({r}, {t}, {f}): self={s}", 7, 
                         r=regex, t=text, f=flags, s=self)
        self.match_result = re.match(regex, text, flags)
        if self.match_result:
            tpo.debug_print("match: %s" % tpo.to_string(self.match_result.groups()), 6)
        return self.match_result

    def get_match(self):
        """Return match result object for last search or match"""
        result = self.match_result
        tpo.debug_format("regex.get_match() => {r}: self={s}", 5, 
                         r=result, s=self)
        return result

    def group(self, num):
        """Return group NUM from match result from last search"""
        debug.assertion(self.match_result)
        result = self.match_result and self.match_result.group(num)
        tpo.debug_format("regex.group({n}) => {r}: self={s}", 5, 
                         n=num, r=result, s=self)
        return result

    def groups(self):
        """Return all groups in match result from last search"""
        debug.assertion(self.match_result)
        result = self.match_result and self.match_result.groups()
        tpo.debug_format("regex.groups() => {r}: self={s}", 5, 
                         r=result, s=self)
        return result

#...............................................................................
# Initialization
#
# note: creates global instance for convenience (and backward compatibility)

my_re = regex()
