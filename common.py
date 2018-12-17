# common.py: Python module with common utilities mostly for debugging
#
# sample usage:
#    from common import *
#    debug_print("start: " + debug_timestamp())
#
#------------------------------------------------------------------------
# TODO:
# - add support for using pprint module
# - add debug_print_value(var, level)
#
#------------------------------------------------------------------------
# Copyright (c) 2012-2018 Thomas P. O'Hara
#

from __future__ import print_function           # enables version 3 print()

# Load required libraries
#
# - For Debugging purposes:
from datetime import datetime
#
# - The usuals:
import sys
import os
import re

# Globals
if __debug__:
    debug_level = 0                     # level at which specific debug tracing occurs
    output_timestamps = False           # prefix output with timestamp (quick-n-dirty profiling)
    output_timestamps_diff = False      # whether to also show time difference from last trace
    last_timestamp = None		# timestamp for keeping track of differentials

#------------------------------------------------------------------------
# Debugging functions
#
# Notes:
# - These are no-op's unless __debug__ is True.
# - Running python with the -O (optimized) option ensures that __debug__ is False.
#

if __debug__:

    # set_debug_level(level): set new debugging level
    #
    def set_debug_level(level):
        global debug_level
        debug_level = level
    

    # debugging_level(): get current debugging level
    #
    def debugging_level():
        global debug_level
        return debug_level
    

    # debug_print_without_newline(level, text): Print TEXT if at debug trace LEVEL or higher,
    # without trailing newline.
    # TODO: work out shorter name (e.g., debug_print_no_eol)
    #
    def debug_print_without_newline(text, level=1):
        if (debug_level >= level):
            if (output_timestamps):
                global timestamp, last_timestamp

                # Get time-proper from timestamp (TODO: find standard way to do this)
                ## OLD: timestamp = debug_timestamp()
                timestamp = datetime.now()
                if output_timestamps_diff and (last_timestamp != None):
                    # Show difference instead with leading 0's in time ignored (e.g., '0:00:05.591210' => '5.591210')
                    timestamp_spec = "+" + re.sub(r"^[0:]+", "", str(timestamp - last_timestamp))
                else:
                    timestamp_spec = re.sub(r"^\d+-\d+-\d+\s*", "", str(timestamp))
                print("[%s] " % timestamp_spec, end='', file=sys.stderr)
                last_timestamp = timestamp
            print(text, end='', file=sys.stderr)
    
    
    # debug_print(level, text): Print TEXT if at debug trace LEVEL or higher.
    # Note: Implemented in terms of debug_print_without_newline to keep timestamp support in one place.
    #
    def debug_print(text, level=1):
        debug_print_without_newline(text, level)
        if (debug_level >= level):
            print(file=sys.stderr)
    

    # debug_timestamp(): Return timestamp for use in debugging traces
    #
    def debug_timestamp():
        return (str(datetime.now()))
    

    # debug_raise(): Raise an exception if debuggig.
    # Note: Intended for use in except clause to produce full stacktrace when debugging.
    # TODO: Have version that just prints complete stacktrace (i.e., without breaking).
    #
    def debug_raise():
        if __debug__:
            raise

else:

    def set_debug_level(level):
        return


    def debugging_level():
        return


    def debug_print_without_newline(text, level=1):
        return


    def debug_print(text, level=1):
        return


    def debug_timestamp():
        return ""


    def debug_raise():
        return


#------------------------------------------------------------------------
# General utility functions


# print_stderr(text): output TEXT to standard error
#
def print_stderr(text):
    print(text, file=sys.stderr)


# getenv_text(var, [default=""]): returns textual value for environment variable VAR (or DEFAULT value)
#
def getenv_text (var, default=""):
    text_value = os.getenv(var)
    if not text_value:
        debug_print("getenv_text: no value for %s" % var, 6)
        text_value = default
    debug_print("getenv_text('%s') => %s" % (var, text_value), 5)
    return (text_value)


# getenv_boolean(var, [default=False]): returns boolean flag based on environment VAR
# (or DEFAULT value)
# Note: "0" or "False" is interpreted as False, and any other value as True.
#
def getenv_boolean (var, default=False):
    bool_value = default
    value_text = getenv_text(var)
    if (len(value_text) > 0):
        bool_value = True
        if (value_text.lower() == "false") or (value_text == "0"):
            bool_value = False
    return (bool_value)


# getenv_number(var, [default=-1]): returns number based on environment VAR (or DEFAULT value).
# getenv_int/getenv_real: typed variants
#
def getenv_number (var, default=-1):
    num_value = default
    value_text = getenv_text(var)
    if (len(value_text) > 0):
        num_value = float(value_text)
    return (num_value)
#
# Variants for integers and real numbers.
#
def getenv_integer (var, default=-1):
    return int(getenv_number(var, default))
#
def getenv_real (var, default=-1):
    return getenv_number(var, default)
#
# Alaises for the above.
#
def getenv_int (var, default=-1):
    return getenv_int(var, default)
#
def getenv_float (var, default=-1):
    return getenv_number(var, default)


# get_current_function_name(): Returns the name of the currently executing function,
# excluding this function of course.
# Based on http://code.activestate.com/recipes/66062-determining-current-function-name.
# Also see http://docs.python.org/2/reference/datamodel.html and http://docs.python.org/2/library/inspect.html.
#
def get_current_function_name():
    name = "???"
    try:
        name = sys._getframe(1).f_code.co_name
    except:
        print_stderr("Exception in get_current_function_name: " + str(sys.exc_info()))
    return name


# get_property_value(object, name, [default=None]):
# EX: import datetime; (get_property_value(datetime.date.today(), 'year', -1) >= 2012) => True
# EX: get_property_value(datetime.date.today(), 'minute', -1) => -1
#
def get_property_value(object, property_name, default_value=None):
    value = default_value
    if (hasattr(object, property_name)):
        value = getattr(object, property_name)
    debug_print("get_property_value%s => %s" % (str((object, property_name, default_value)), 
                                                value), level=4)
    return value

#------------------------------------------------------------------------
# Memomization support (i.e., functiona result caching), based on 
#     See http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-. [world!]
# This is implemented transparently via Python decorators. See
#     http://stackoverflow.com/questions/739654/understanding-python-decorators
#
# usage example:
#
#    @memodict
#    def fubar(word):
#        result = ...
#        return (result)
#

def memodict(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret 
    return memodict().__getitem__


#------------------------------------------------------------------------

# Override debug level to DEBUG_LEVEL environment variable unless running in optimized mode (i.e., __debug__ is False)
if __debug__:
    env_debug_level = os.getenv("DEBUG_LEVEL")
    debug_level = int(env_debug_level) if env_debug_level else 1
    output_timestamps_diff = getenv_boolean("OUTPUT_DEBUG_TIMESTAMPS_DIFF", output_timestamps_diff)
    output_timestamps = getenv_boolean("OUTPUT_DEBUG_TIMESTAMPS", output_timestamps_diff)

    # Show trace level in effect
    debug_print("common.py: debug_level=%d" % debug_level, level=3)

# Warn if invoked standalone
#
if __name__ == '__main__':
    print_stderr("Warning: common.py is not intended to be run standalone")
