# Convenience class for regex searching, providing simple wrapper around
# static match results.
#
# Example usage:
#
#    from my_regex import my_re
#    ...
#    if (my_re.search(r"^(\d+)\:?(\d*)\s+(\d+)\:?(\d*)\s+\S.*", line)):
#        (start_hours, start_mins, end_hours, end_mins) = my_re.groups()
#
#--------------------------------------------------------------------------------
# TODO: weed out tpo_common usages (work out converter script for common cases like tpo.debug_print to debug.trace and tpo.debug_format to debug.trace_fmt)
# TODO:
# - Add examples for group(), groups(), etc.
# - Clean up script (e.g., regex => regex_wrapper).
# - Add perl-inspired accessors (e.g., PREMATCH, POSTMATCH).
#
#

"""Wrapper class for regex match results"""

import re
import six

import debug
import tpo_common as tpo

# Expose public symbols from re package
__all__ = re.__all__
## TODO # HACK: make sure regex can be used as plug-in replacement 
## from from re import *

class regex_wrapper(object):
    """Wrapper class over re to implement regex search that saves match results
    note: Allows regex to be used directly in conditions"""
    # TODO: IGNORECASE = re.IGNORECASE, etc.
    # import from RE so other methods supported directly (and above constants)

    def __init__(self, ):
        tpo.debug_format("regex.__init__(): self={s}", 4, s=self)
        self.match_result = None
        # TODO: self.regex = ""

    def search(self, regex, text, flags=0):
        """Search for REGEX in TEXT with optional FLAGS"""
        tpo.debug_format("regex.search({r}, {t}, {f}): self={s}", 7,
                         r=regex, t=text, f=flags, s=self)
        debug.assertion(isinstance(text, six.string_types))
        self.match_result = re.search(regex, text, flags)
        if self.match_result:
            ## OLD: tpo.debug_print("match: %s" % tpo.to_string(self.match_result.groups()), 6)
            debug.trace_fmt(6, "match: {m}; regex: {r}", m=self.grouping(), r=regex)
        return self.match_result

    def match(self, regex, text, flags=0):
        """Match REGEX to TEXT with optional FLAGS"""
        tpo.debug_format("regex.match({r}, {t}, {f}): self={s}", 7,
                         r=regex, t=text, f=flags, s=self)
        self.match_result = re.match(regex, text, flags)
        if self.match_result:
            ## OLD: tpo.debug_print("match: %s" % tpo.to_string(self.match_result.groups()), 6)
            debug.trace_fmt(6, "match: {m}; regex: {r}", m=self.grouping(), r=regex)
        return self.match_result

    def get_match(self):
        """Return match result object for last search or match"""
        result = self.match_result
        tpo.debug_format("regex.get_match() => {r}: self={s}", 7,
                         r=result, s=self)
        return result

    def group(self, num):
        """Return group NUM from match result from last search"""
        debug.assertion(self.match_result)
        result = self.match_result and self.match_result.group(num)
        tpo.debug_format("regex.group({n}) => {r}: self={s}", 7,
                         n=num, r=result, s=self)
        return result

    def groups(self):
        """Return all groups in match result from last search"""
        debug.assertion(self.match_result)
        result = self.match_result and self.match_result.groups()
        debug.trace_fmt(7, "regex.groups() => {r}: self={s}",
                        r=result, s=self)
        return result

    def grouping(self):
        """Return groups for match result or entire matching string if no groups defined"""
        # Note: this is intended to facilitate debug tracing; see example in search method above
        result = self.match_result and (self.match_result.groups() or self.match_result.group(0))
        debug.trace_fmt(7, "regex.grouping() => {r}: self={s}", r=result, s=self)
        return result

    def start(self, group=0):
        """Start index for GROUP"""
        result = self.match_result and self.match_result.start(group)
        debug.trace_fmt(7, "regex.start({g}) => {r}: self={s}", r=result, s=self, g=group)
        return result

    def end(self, group=0):
        """End index for GROUP"""
        result = self.match_result and self.match_result.end(group)
        debug.trace_fmt(7, "regex.end({g}) => {r}: self={s}", r=result, s=self, g=group)
        return result

#...............................................................................
# Initialization
#
# note: creates global instance for convenience (and backward compatibility)

my_re = regex_wrapper()
