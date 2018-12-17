#! /usr/bin/env python
#
# Functions for debugging, such as console tracing. This is for intended
# for verbose tracing not suitable for the logging facility.
#
# Notes:
# - These are no-op's unless __debug__ is True.
# - Running python with the -O (optimized) option ensures that __debug__ is False.
# - So that other local packages can use tracing freely, this only
#   imports standard packages. In particular, system.py is not imported,
#   so functionality must be reproduced here (e.g., _to_utf8).
#
# TODO:
# - Rename as debug_utils so clear that non-standard package.
# - Add my_assert that isses error message rather than raising exception.
#
# Copyright (c) 2012-2018 Thomas P. O'Hara
#

"""Debugging functions (e.g., tracing)"""

## OLD: if sys.version_info.major == 2:
## OLD:    from __future__ import print_function
from __future__ import print_function
import sys_version_info_hack

# Standard packages
import atexit
from datetime import datetime
import inspect
import os
from pprint import pprint
import re
import sys

ALWAYS = 0
ERROR = 1
WARNING = 2
USUAL = 3
DETAILED = 4
VERBOSE = 5

if __debug__:    

    # Initialize debug tracing level
    DEBUG_LEVEL_LABEL = "DEBUG_LEVEL"
    trace_level = 1
    output_timestamps = False
    #
    try:
        trace_level = int(os.environ.get(DEBUG_LEVEL_LABEL, trace_level))
    except:
        sys.stderr.write("Warning: Unable to set tracing level from {v}: {exc}\n".
                         format(v=DEBUG_LEVEL_LABEL, exc=sys.exc_info()))


    def set_level(level):
        """Set new trace level"""
        global trace_level
        trace_level = level
        return


    def get_level():
        """Get current tracing level"""
        ## global trace_level
        return trace_level


    def get_output_timestamps():
        """Return whether outputting timestamps"""
        return output_timestamps


    def set_output_timestamps(do_output_timestamps):
        """Enable for disable the outputting of timestamps"""
        global output_timestamps
        output_timestamps = do_output_timestamps


    def _to_utf8(text):
        """Convert TEXT to UTF-8 (e.g., for I/O)"""
        # Note: version like one from system.py to avoid circular dependency
        result = text
        if ((sys.version_info.major < 3) and (isinstance(text, unicode))):
            result = result.encode("UTF-8", 'ignore')
        return result


    def trace(level, text):
        """Print TEXT if at trace LEVEL or higher, including newline"""
        if (trace_level >= level):
            # Prefix trace with timestamp
            if output_timestamps:
                # Get time-proper from timestamp (TODO: find standard way to do this)
                timestamp = re.sub(r"^\d+-\d+-\d+\s*", "", debug_timestamp())

                print("[" + timestamp + "]", end=": ", file=sys.stderr)
            # Print trace, converted to UTF8 if necessary (Python2 only)
            print(_to_utf8(text), file=sys.stderr)
        return


    def trace_fmtd(level, text, **kwargs):
        """Print TEXT with formatting using optional format KWARGS if at trace LEVEL or higher, including newline"""
        if (trace_level >= level):
            try:
                trace(level, text.format(**kwargs))
            except(KeyError, ValueError, UnicodeEncodeError):
                sys.stderr.write("Warning: Problem in trace_fmtd: {exc}\n".
                                 format(exc=sys.exc_info()))
                # Show arguments so trace contents recoverable
                sys.stderr.write("   text=%r\n" % _to_utf8(clip_value(text)))
                kwargs_spec = ", ".join(("%s:%r" % (k, clip_value(v))) for (k, v) in kwargs.iteritems())
                sys.stderr.write("   kwargs=%s\n" % _to_utf8(kwargs_spec))
        return


    def trace_object(level, obj, label=None, show_all=False, indentation=None, pretty_print=None):
        """Trace out OBJ's members to stderr if at trace LEVEL or higher"""
        # Note: This is intended for arbitrary objects, use trace_values for lists or hashes.
        # See https://stackoverflow.com/questions/383944/what-is-a-python-equivalent-of-phps-var-dump.
        # TODO: support recursive trace
        ## OLD: print("{stmt} < {current}: {r}".format(stmt=level, current=trace_level,
        ##                                       r=(trace_level < level)))
        if (pretty_print is None):
            pretty_print = (trace_level >= 6)
        if (trace_level < level):
            return
        if label is None:
            label = str(type(obj)) + " " + hex(hash(obj))
        if indentation is None:
            indentation = "   "
        trace(0, label + ": {")
        for (member, value) in inspect.getmembers(obj):
            # TODO: value = clip_text(value)
            trace_fmtd(8, "{i}{m}={v}; type={t}", i=indentation, m=member, v=value, t=type(value))
            if (trace_level >= 9):
                ## print(indentation + member + ":", value, file=sys.stderr)
                sys.stderr.write(indentation + member + ":")
                if pretty_print:
                    pprint(value, stream=sys.stderr)
                else:
                    sys.stderr.write(value, file=sys.stderr)
                sys.stderr.write("\n")
                continue
            ## TODO: pprint.pprint(member, stream=sys.stderr, indent=4, width=512)
            try:
                value_spec = "%s" % ((value),)
            except(TypeError, ValueError):
                trace_fmtd(7, "Warning: Problem in tracing member {m}: {exc}",
                           m=member, exc=sys.exc_info())
                value_spec = "__n/a__"
            if (show_all or (not (member.startswith("__") or 
                                  re.search('^<.*(method|module|function).*>$', value_spec)))):
                ## trace(0, indentation + member + ": " + value_spec)
                sys.stderr.write(indentation + member + ": ")
                if pretty_print:
                    # TODO: remove quotes from numbers and booleans
                    pprint(value_spec, stream=sys.stderr, indent=len(indentation))
                else:
                    sys.stderr.write(value_spec)
                    sys.stderr.write("\n")
        trace(0, indentation + "}")
        return


    def trace_values(level, collection, label=None, indentation=None):
        """Trace out elements of array or hash COLLECTION if at trace LEVEL or higher"""
        assert(isinstance(collection, (list, dict)))
        if (trace_level < level):
            return
        if indentation is None:
            indentation = "   "
        if label is None:
            label = str(type(collection)) + " " + hex(hash(collection))
            indentation = "   "
        trace(0, label + ": {")
        keys_iter = collection.iterkeys() if isinstance(collection, dict) else xrange(len(collection))
        for k in keys_iter:
            try:
                trace_fmtd(0, "{ind}{k}: {v}", ind=indentation, k=k,
                           v=_to_utf8(collection[k]))
            except:
                trace_fmtd(7, "Warning: Problem tracing item {k}",
                           k=_to_utf8(k), exc=sys.exc_info())
        trace(0, indentation + "}")
        return


    def debug_timestamp():
        """Return timestamp for use in debugging traces"""
        return (str(datetime.now()))


    def raise_exception(level=1):
        """Raise an exception if debugging (at specified trace LEVEL)"""
        # Note: For producing full stacktrace in except clause when debugging.
        if __debug__ and (level <= trace_level):
            raise                       # pylint: disable=misplaced-bare-raise
        return


    def assertion(expression):
        """Issue warning if EXPRESSION doesn't hold"""
        # EX: assertion((2 + 2) != 5)
        if (not expression):
            try:
                # Get source information for failed assertion
                trace_fmtd(9, "Call stack: {st}", st=inspect.stack())
                caller = inspect.stack()[1]
                (_frame, filename, line_number, _function, _context, _index) = caller
                # Read statement in file and extract assertion expression
                # TODO: handle #'s in statement proper (e.g., assertion("#" in text))
                statement = read_line(filename, line_number).strip()
                statement = re.sub("#.*$", "", statement)
                statement = re.sub(r"^(\S*)assertion\(", "", statement)
                expression = re.sub(r"\);?\s*$", "", statement)
                # Output information
                trace_fmtd(0, "Assertion failed: {expr} (at {file}:{line})",
                           expr=expression, file=filename, line=line_number)
            except:
                trace_fmtd(0, "Exception formatting assertion: {exc}",
                           exc=sys.exc_info())
                trace_object(0, inspect.currentframe(), "caller frame", pretty_print=True)
        return


    ## TODO: output_timestamps = getenv_boolean("OUTPUT_DEBUG_TIMESTAMPS", output_timestamps)
    output_timestamps = (str(os.environ.get("OUTPUT_DEBUG_TIMESTAMPS", False)).upper()
                         in ["1", "TRUE"])

    # Show startup time and tracing info
    MODULE_FILE = __file__
    trace_fmtd(3, "[{f}] loaded at {t}", f=MODULE_FILE, t=debug_timestamp())
    trace_fmtd(4, "trace_level={l}; output_timestamps={ots}", l=trace_level, ots=output_timestamps)

    # Register to show shuttdown time
    atexit.register(lambda: trace_fmtd(3, "[{f}] unloaded at {t}", f=MODULE_FILE, t=debug_timestamp()))
    
else:

    def non_debug_stub(*_args, **_kwargs):
        """Non-debug stub"""
        pass


    def get_level():
        """Returns tracing level (i.e., 0)"""
        return 0


    def get_output_timestamps():
        """Non-debug stub"""
        return False


    set_output_timestamps = non_debug_stub


    trace = non_debug_stub


    trace_fmtd = non_debug_stub


    trace_object = non_debug_stub


    debug_timestamp = non_debug_stub


    raise_exception = non_debug_stub


    assertion = non_debug_stub

def debugging(level=ERROR):
    """Whether debugging at specified trace level, which defaults to {l}""".format(l=ERROR)
    return (get_level() >= level)
    
def detailed_debugging():
    """Whether debugging with trace level at or above {l}""".format(l=DETAILED)
    return (get_level() >= DETAILED)

def verbose_debugging():
    """Whether debugging with trace level at or above {l}""".format(l=VERBOSE)
    return (get_level() >= VERBOSE)

#-------------------------------------------------------------------------------
# Utility functions useful for debugging (e.g., for trace output)

# TODO: CLIPPED_MAX = system.getenv_int("CLIPPED_MAX", 132)
CLIPPED_MAX = 132
#
def clip_value(value):
    """Return clipped version of VALUE (e.g., first 132 chars)"""
    # TODO: omit conversion to text if already text [DUH!]
    clipped = "%s" % value
    if (len(clipped) > CLIPPED_MAX):
        clipped = clipped[:CLIPPED_MAX] + "..."
    return clipped

def read_line(filename, line_number):
    """Returns contents of FILENAME at LINE_NUMBER"""
    # ex: "debugging" in read_line(os.path.join(os.getcwd(), "debug.py"), 3)
    try:
        file_handle = open(filename)
        line_contents = (list(file_handle))[line_number - 1]
    except:
        line_contents = "???"
    return line_contents

#-------------------------------------------------------------------------------

def main(_args):
    """Supporting code for command-line processing"""
    trace(1, "Warning: Not intended for direct invocation. A simple tracing example follows.")
    trace_object(1, datetime.now(), label="now")
    # TODO: debug.assertion(2 + 2 == 5)
    return


if __name__ == '__main__':
    main(sys.argv)
