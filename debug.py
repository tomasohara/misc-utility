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
# - Gotta hate Pythonista's who prevailed in Python3 breaking lots of Python2
#   code just for the sake of simplicity, a la manera moronista (i.e., only one moronic way to do things)!
#
# TODO:
# - Rename as debug_utils so clear that non-standard package.
# - Add exception handling throughout (e.g., more in trace_object).
#

"""Debugging functions (e.g., tracing)"""

## OLD: if sys.version_info.major == 2:
## OLD:    from __future__ import print_function
from __future__ import print_function

# Standard packages
import atexit
from datetime import datetime
import inspect
import os
from pprint import pprint
import re
import six
import sys
import time

# Local packages
# note: The following redefines sys.version_info to be python3 compatible;
# this is used in _to_utf8, which should be reworked via six-based wrappers.
import sys_version_info_hack              # pylint: disable=unused-import


# Constants for pre-defined tracing levels
ALWAYS = 0
ERROR = 1
WARNING = 2
USUAL = 3
DETAILED = 4
VERBOSE = 5
QUITE_DETAILED = 6
QUITE_VERBOSE = 7
MOST_DETAILED = 8
MOST_VERBOSE = 9

# Other constants
UTF8 = "UTF-8"
STRING_TYPES = six.string_types

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
        if ((sys.version_info.major < 3) and (isinstance(text, unicode))):  # pylint: disable=undefined-variable
            result = result.encode("UTF-8", 'ignore')
        return result


    def _to_unicode(text, encoding=None):
        """Ensure TEXT in ENCODING is Unicode, such as from the default UTF8"""
        # TODO: rework from_utf8 in terms of this
        if not encoding:
            encoding = UTF8
        result = text
        if ((sys.version_info.major < 3) and (not isinstance(result, unicode))): # pylint: disable=undefined-variable
            result = result.decode(encoding, 'ignore')
        return result


    def _to_string(text):
        """Ensure TEXT is a string type"""
        result = text
        if (not isinstance(result, STRING_TYPES)):
            # Values are coerced using % operator for proper Unicode handling,
            # except for tuples which are converted recursively. This avoids a
            # type error due to arguments not being converted (e.g., second 
            # tuple constituent, etc.), as in ("%s" % (9, 1)).
            if isinstance(result, tuple):
                result = "(" + ", ".join([_to_string(v) for v in result]) + ")"
            else:
                result = "%s" % result
        return result

    
    def trace(level, text):
        """Print TEXT if at trace LEVEL or higher, including newline"""
        # Note: trace should not be used with text that gets formatted to avoid
        # subtle errors
        if (trace_level >= level):
            # Prefix trace with timestamp w/o date
            if output_timestamps:
                # Get time-proper from timestamp (TODO: find standard way to do this)
                timestamp_time = re.sub(r"^\d+-\d+-\d+\s*", "", timestamp())

                print("[" + timestamp_time + "]", end=": ", file=sys.stderr)
            # Print trace, converted to UTF8 if necessary (Python2 only)
            # TODO: add version of assertion that doesn't use trace or trace_fmtd
            ## TODO: assertion(not(re.search(r"{\S*}", text)))
            print(_to_utf8(text), file=sys.stderr)
        return


    def trace_fmtd(level, text, **kwargs):
        """Print TEXT with formatting using optional format KWARGS if at trace LEVEL or higher, including newline"""
        # Note: To avoid interpolated text as being interpreted as variable
        # references, this function does the formatting.
        # TODO: weed out calls that use (level, text.format(...)) rather than (level, text, ...)
        if (trace_level >= level):
            try:
                # TODO: add version of assertion that doesn't use trace or trace_fmtd
                ## TODO: assertion(re.search(r"{\S*}", text))
                ## OLD: assertion("{" in text)
                ## OLD: trace(level, text.format(**kwargs))
                kwargs_unicode = {k:_to_unicode(_to_string(v)) for (k, v) in list(kwargs.items())}
                trace(level, _to_unicode(text).format(**kwargs_unicode))
            except(KeyError, ValueError, UnicodeEncodeError):
                raise_exception(max(VERBOSE, level + 1))
                sys.stderr.write("Warning: Problem in trace_fmtd: {exc}\n".
                                 format(exc=sys.exc_info()))
                # Show arguments so trace contents recoverable
                sys.stderr.write("   text=%r\n" % _to_utf8(clip_value(text)))
                ## OLD: kwargs_spec = ", ".join(("%s:%r" % (k, clip_value(v))) for (k, v) in kwargs.iteritems())
                kwargs_spec = ", ".join(("%s:%r" % (k, clip_value(v))) for (k, v) in list(kwargs.items()))
                sys.stderr.write("   kwargs=%s\n" % _to_utf8(kwargs_spec))
        return


    def trace_object(level, obj, label=None, show_all=False, indentation=None, pretty_print=None):
        """Trace out OBJ's members to stderr if at trace LEVEL or higher"""
        # Note: This is intended for arbitrary objects, use trace_values for lists or hashes.
        # See https://stackoverflow.com/questions/383944/what-is-a-python-equivalent-of-phps-var-dump.
        # TODO: support recursive trace; specialize show_all into show_private and show_methods
        ## OLD: print("{stmt} < {current}: {r}".format(stmt=level, current=trace_level,
        ##                                       r=(trace_level < level)))
        if (pretty_print is None):
            pretty_print = (trace_level >= 6)
        if (trace_level < level):
            return
        if label is None:
            ## BAD: label = str(type(obj)) + " " + hex(hash(obj))
            label = str(type(obj)) + " " + hex(id(obj))
        if indentation is None:
            indentation = "   "
        trace(0, label + ": {")
        ## OLD: for (member, value) in inspect.getmembers(obj):
        member_info = []
        try:
            member_info = inspect.getmembers(obj)
        except:
            trace_fmtd(7, "Warning: Problem getting member list in trace_object: {exc}",
                       exc=sys.exc_info())
        for (member, value) in member_info:
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
                                  re.search(r"^<.*(method|module|function).*>$", value_spec)))):
                ## trace(0, indentation + member + ": " + value_spec)
                sys.stderr.write(indentation + member + ": ")
                if pretty_print:
                    # TODO: remove quotes from numbers and booleans
                    pprint(value_spec, stream=sys.stderr, indent=len(indentation))
                else:
                    ## sys.stderr.write(value_spec)
                    sys.stderr.write(_to_utf8(value_spec))
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
            ## BAD: label = str(type(collection)) + " " + hex(hash(collection))
            label = str(type(collection)) + " " + hex(id(collection))
            indentation = "   "
        trace(0, label + ": {")
        ## OLD: keys_iter = collection.iterkeys() if isinstance(collection, dict) else xrange(len(collection))
        ## OLD2: keys_iter = collection.iterkeys() if isinstance(collection, dict) else range(len(collection))
        keys_iter = list(collection.keys()) if isinstance(collection, dict) else range(len(collection))
        ## NOTE: Gotta hate python3 for dropping xrange [la manera moronista!]
        for k in keys_iter:
            try:
                trace_fmtd(0, "{ind}{k}: {v}", ind=indentation, k=k,
                           v=_to_utf8(collection[k]))
            except:
                trace_fmtd(7, "Warning: Problem tracing item {k}",
                           k=_to_utf8(k), exc=sys.exc_info())
        trace(0, indentation + "}")
        return


    def trace_current_context(level=QUITE_DETAILED, label=None, 
                              show_methods_etc=False):
        """Traces out current context (local and global variables), with output
        prefixed by "LABEL context" (e.g., "current context: {\nglobals: ...}").
        Notes: By default the debugging level must be quite-detailed (6).
        If the debugging level is higher, the entire stack frame is traced.
        Also, methods are omitted by default."""
        frame = None
        if label is None:
            label = "current"
        try:
            frame = inspect.currentframe().f_back
        except (AttributeError, KeyError, ValueError):
            trace_fmt(5, "Exception during trace_current_context: {exc}",
                      exc=sys.exc_info())
        trace_fmt(level, "{label} context: {{", label=label)
        prefix = "    "
        if (get_level() - level) > 1:
            trace_object((level + 2), frame, "frame", indentation=prefix,
                         show_all=show_methods_etc)
        else:
            trace_fmt(level, "frame = {f}", f=frame)
            if frame:
                trace_object(level, frame.f_globals, "globals", indentation=prefix,
                             show_all=show_methods_etc)
                trace_object(level, frame.f_locals, "locals", indentation=prefix,
                             show_all=show_methods_etc)
        trace(level, "}")
        return


    def raise_exception(level=1):
        """Raise an exception if debugging (at specified trace LEVEL)"""
        # Note: For producing full stacktrace in except clause when debugging.
        if __debug__ and (level <= trace_level):
            raise                       # pylint: disable=misplaced-bare-raise
        return


    def assertion(expression):
        """Issue warning if EXPRESSION doesn't hold"""
        ## TODO: have streamlined version using sys.write that can be used for trace and trace_fmtd sanity checks about {}'s
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


else:

    def non_debug_stub(*_args, **_kwargs):
        """Non-debug stub"""
        return


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
    
    trace_current_context = non_debug_stub
   
    raise_exception = non_debug_stub
    
    assertion = non_debug_stub

    
# note: adding alias for trace_fmtd to account for common typo
# TODO: alias trace to trace_fmt as well (add something like trace_out if format not desired)
trace_fmt = trace_fmtd

def debug_print(text, level):
    """Wrapper around trace() for backward compatibility
    Note: debug_print will soon be deprecated."""
    return trace(level, text)


def timestamp():
    """Return timestamp for use in logging, etc."""
    return (str(datetime.now()))
    

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

# TODO: CLIPPED_MAX = system-ish.getenv_int("CLIPPED_MAX", 132)
CLIPPED_MAX = 132
#
def clip_value(value, max_len=CLIPPED_MAX):
    """Return clipped version of VALUE (e.g., first MAX_LEN chars)"""
    # TODO: omit conversion to text if already text [DUH!]
    clipped = "%s" % value
    if (len(clipped) > max_len):
        clipped = clipped[:max_len] + "..."
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
    return

# Do debug-only processing (n.b., for when PYTHONOPTIMIZE not set)
# Note: wrapped in function to keep things clean

if __debug__:

    def debug_init():
        """Debug-only initialization"""
        time_start = time.time()
        trace(4, "in debug_init()")
        
        # Determine whether tracing include time and date
        global output_timestamps
        output_timestamps = (str(os.environ.get("OUTPUT_DEBUG_TIMESTAMPS", False)).upper()
                             in ["1", "TRUE"])
    
        # Show startup time and tracing info
        module_file = __file__
        trace_fmtd(3, "[{f}] loaded at {t}", f=module_file, t=timestamp())
        trace_fmtd(4, "trace_level={l}; output_timestamps={ots}", l=trace_level, ots=output_timestamps)
    
        # Register to show shuttdown time and elapsed seconds
        def display_ending_time_etc():
            """Display ending time information"""
            elapsed = round(time.time() - time_start, 3)
            trace_fmtd(4, "[{f}] unloaded at {t}; elapsed={e}s",
                       f=module_file, t=timestamp(), e=elapsed)
        atexit.register(display_ending_time_etc)
        
        return

    
    # Do the initialization
    debug_init()

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    main(sys.argv)
