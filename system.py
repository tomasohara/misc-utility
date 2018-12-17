#! /usr/bin/env python
#
# Functions for system related access, such as running command or
# getting environment values.
#
# TODO:
# - Rename as system_utils so clear that non-standard package.
# - Add support for maintaining table of environment variables with optional descriptions.
#
# Copyright (c) 2012-2018 Thomas P. O'Hara
#

"""System-related functions"""

## OLD: if sys.version_info.major == 2:
## OLD:     from __future__ import print_function
from __future__ import print_function

# Standard packages
import inspect
import os
import pickle
import re
import sys
import types
import urllib

# Local packages
import debug


def print_stderr(text, **kwargs):
    """Output TEXT to standard error"""
    # TODO: add support for format keywords (e.g., print_stderr("Error: fubar={f}", f=fubar)
    formatted_text = text
    try:
        formatted_text = text.format(**kwargs)
    except(KeyError, ValueError, UnicodeEncodeError):
        sys.stderr.write("Warning: Problem in print_stderr: {exc}\n".
                                 format(exc=sys.exc_info()))
    print(formatted_text, file=sys.stderr)
    return


def getenv_text(var, default="", _description=None):
    """Returns textual value for environment variable VAR (or DEFAULT value)"""
    text_value = os.getenv(var)
    if not text_value:
        debug.trace_fmtd(6, "getenv_text: no value for var {v}", v=var)
        text_value = default
    debug.trace_fmtd(5, "getenv_text('{v}',  [{d}]) => {r}",
                     v=var, d=default, r=text_value)
    return (text_value)


def getenv_bool(var, default=False, _description=None):
    """Returns boolean flag based on environment VAR (or DEFAULT value)"""
    # Note: "0" or "False" is interpreted as False, and any other value as True.
    bool_value = default
    value_text = getenv_text(var)
    if (len(value_text) > 0):
        bool_value = to_bool(value_text)
    debug.trace_fmtd(6, "getenv_bool({v}, {d}) => {r}",
                     v=var, d=default, r=bool_value)
    return (bool_value)
#
getenv_boolean = getenv_bool


def getenv_number(var, default=-1.0, _description=None):
    """Returns number based on environment VAR (or DEFAULT value)."""
    # Note: use getenv_int or getenv_float for typed variants
    num_value = default
    value_text = getenv_text(var)
    if (len(value_text) > 0):
        num_value = float(value_text)
    debug.trace_fmtd(6, "getenv_number({v}, {d}) => {r}",
                     v=var, d=default, r=num_value)
    return (num_value)


def getenv_int(var, default=-1, _description=None):
    """Version of getenv_number for integers"""
    value = getenv_number(var, default)
    if (value is not None):
        value = int(value)
    return (value)
#
getenv_integer = getenv_int


def getenv_float(var, default=-1, _description=None):
    """Version of getenv_number for floating point values"""
    return (getenv_number(var, default))


def setenv(var, value):
    """Set environment VAR to VALUE"""
    debug.trace_fmtd(6, "getenv({v}, {val})", v=var, val=value)
    os.environ[var] = value
    return


def print_full_stack(stream=sys.stderr):
    """Prints stack trace (for use in error messages, etc.)"""
    # Notes: Developed originally for Android stack tracing support.
    # Based on http://blog.dscpl.com.au/2015/03/generating-full-stack-traces-for.html.
    # TODO: Update based on author's code update (e.g., ???)
    # TODO: Fix off-by-one error in display of offending statement!
    debug.trace_fmtd(7, "print_full_stack(stream={s})", s=stream)
    stream.write("Traceback (most recent call last):\n")
    try:
        # Note: Each tuple has the form (frame, filename, line_number, function, context, index)
        item = None
        # Show call stack excluding caller
        for item in reversed(inspect.stack()[2:]):
            stream.write('  File "{1}", line {2}, in {3}\n'.format(*item))
        for line in item[4]:
            stream.write('  ' + line.lstrip())
        # Show context of the exception from caller to offending line
        stream.write("  ----------\n")
        for item in inspect.trace():
            stream.write('  File "{1}", line {2}, in {3}\n'.format(*item))
        for line in item[4]:
            stream.write('  ' + line.lstrip())
    except:
        debug.trace_fmtd(3, "Unable to produce stack trace: {exc}", exc=sys.exc_info())
    stream.write("\n")
    return


def save_object(file_name, obj):
    """Saves OBJ to FILE_NAME in pickle format"""
    # Note: The data file is created in binary mode to avoid quirk under Windows.
    # See https://stackoverflow.com/questions/556269/importerror-no-module-named-copy-reg-pickle.
    debug.trace_fmtd(6, "save_object({f}, _)", f=file_name)
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f)
    except (IOError, TypeError, ValueError):
        debug.trace_fmtd(1, "Error: Unable to save object to {f}: {exc}",
                         f=file_name, exc=sys.exc_info())
    return

    
def load_object(file_name, ignore_error=False):
    """Loads object from FILE_NAME in pickle format"""
    obj = None
    try:
        with open(file_name, 'r') as f:
            obj = pickle.load(f)
    except (IOError, ValueError):
        if (not ignore_error):
            print_stderr("Error: Unable to load object from {f}: {exc}".
                         format(f=file_name, exc=sys.exc_info()))
    debug.trace_fmtd(7, "load_object({f}) => {o}", f=file_name, o=obj)
    return obj


def quote_url_text(text):
    """Quote TEXT to make suitable for use in URL. Note: This return the input if the text has encoded characters (i.e., %HH) where H is uppercase hex digit."""
    # Note: This is a wrapper around quote_plus and thus escapes slashes, along with spaces and other special characters (";?:@&=+$,\"'").
    # EX: quote_url_text("<2/") => "%3C2%2f"
    # EX: quote_url_text("Joe's hat") => "Joe%27s+hat"
    # EX: quote_url_text("Joe%27s+hat") => "Joe%27s+hat"
    debug.trace_fmtd(7, "in quote_url_text({t})", t=text)
    result = text
    if not re.search("%[0-9A-F]{2}", text):
        if sys.version_info.major > 2:
            result = urllib.parse.quote_plus(text)
        else:
            result = urllib.quote_plus(to_utf8(text))
    debug.trace_fmtd(6, "out quote_url_text({t}) => {r}", t=text, r=result)
    return result


def escape_html_text(text):
    """Add entity encoding to TEXT to make suitable for HTML"""
    # Note: This is wrapper around html.escape and just handles
    # '&', '<', '>', and '"'.
    # EX: escape_html_text("<2/") => "&lt;2/"
    # EX: escape_html_text("Joe's hat") => "Joe's hat"
    debug.trace_fmtd(7, "in escape_html_text({t})", t=text)
    result = ""
    if sys.version_info.major > 2:
        import html
        result = html.escape(text)
    else:
        import cgi
        ## result = cgi.escape(to_utf8(text))
        result = cgi.escape(text, quote=True)
    debug.trace_fmtd(6, "out escape_html_text({t}) => {r}", t=text, r=result)
    return result


def read_entire_file(filename):
    """Read all of FILENAME and return as a string"""
    data = ""
    try:
        with open(filename) as f:
            data = from_utf8(f.read())
    except IOError:
        debug.trace_fmtd(1, "Error: Unable to read file '{f}': {exc}",
                         f=filename, exc=sys.exc_info())
    debug.trace_fmtd(7, "read_entire_file({f}) => {r}", f=filename, r=data)
    return data


def read_lookup_table(filename):
    """Reads FILENAME and returns as hash lookup"""
    hash_table = {}
    try:
        with open(filename) as f:
            for line in f:
                line = from_utf8(line)
                # TODO: trap exception and lowercase key
                (key, value) = line.split("\t", 1)
                hash_table[key] = value
            ## BAD: hash_table = from_utf8(f.read())
    except (IOError, ValueError):
        debug.trace_fmtd(1, "Error: Creating lookup from '{f}': {exc}",
                         f=filename, exc=sys.exc_info())
    debug.trace_fmtd(7, "read_lookup_table({f}) => {r}", f=filename, r=hash_table)
    return hash_table


def write_file(filename, text):
    """Create FILENAME with TEXT"""
    with open(filename, "w") as f:
        try:
            ## OLD: f.write(to_utf8(text) + "\n")
            f.write(to_utf8(text))
            if not text.endswith("\n"):
                f.write("\n")
        except (IOError, ValueError):
            debug.trace_fmtd(1, "Error: Problem writing file '{f}': {exc}",
                             f=filename, exc=sys.exc_info())
    return


def remove_extension(filename):
    """Return FILENAME without final extension"""
    # ex: remove_extension("document.pdf") => "document")
    new_filename = re.sub(r"\.[^\.]*$", "", filename)
    debug.trace_fmtd(4, "remove_extension({f}) => {r}", f=filename, r=new_filename)
    return new_filename


def get_file_size(filename):
    """Returns size of FILENAME or -1 if not found"""
    size = -1
    if os.path.exists(filename):
        size = os.path.getsize(filename)
    debug.trace_fmtd(5, "get_file_size({f}) => {s}", f=filename, s=size)
    return size


def download_web_document(url, filename=None, meta_hash=None):
    """Download document contents at URL, returning as unicode text. An optional FILENAME can be given for the download, and an optional META_HASH can be specified for recording filename and headers"""
    debug.trace_fmtd(4, "download_web_document({u}, {f}, {mh})", u=url, f=filename, mh=meta_hash)
    # EX: "currency" in download_web_document("https://simple.wikipedia.org/wiki/Dollar")

    # Download the document and optional headers (metadata).
    # Note: urlretrieve chokes on URLS like www.cssny.org without the protocol.
    # TODO: report as bug if not fixed in Python 3
    if filename is None:
        filename = quote_url_text(url)
        debug.trace_fmtd(5, "\tquoted filename: {f}", f=filename)
    if "//" not in url:
        url = "http://" + url
    local_filename = filename
    headers = ""
    if non_empty_file(local_filename):
        debug.trace_fmtd(5, "Using cached file for URL: {f}", f=local_filename)
    else:
        try:
            if sys.version_info.major > 2:
                local_filename, headers = urllib.request.urlretrieve(url, filename)
            else:
                local_filename, headers = urllib.urlretrieve(url, filename)
            debug.trace_fmtd(5, "=> local file: {f}; headers={{h}}",
                             f=local_filename, h=headers)
        except IOError:
            debug.raise_exception(6)
            debug.trace_fmtd(1, "Error: Unable to download {u}: {exc}",
                             u=url, exc=sys.exc_info())
    if meta_hash is not None:
        meta_hash["filename"] = local_filename
        meta_hash["headers"] = headers

    # Read all of the data and return as text
    data = read_entire_file(local_filename)
    debug.trace_fmtd(7, "download_document() => {d}", d=data)
    return data


def to_utf8(text):
    """Convert TEXT to UTF-8 (e.g., for I/O)"""
    result = text
    if ((sys.version_info.major < 3) and (isinstance(text, unicode))):
        result = result.encode("UTF-8", 'ignore')
    debug.trace_fmtd(8, "to_utf8({t}) => {r}", t=text, r=result)
    return result


def from_utf8(text):
    """Convert TEXT to Unicode from UTF-8"""
    result = text
    if ((sys.version_info.major < 3) and (not isinstance(text, unicode))):
        result = result.decode("UTF-8", 'ignore')
    debug.trace_fmtd(8, "from_utf8({t}) => {r}", t=text, r=result)
    return result


def to_string(text):
    """Ensure TEXT is a string type"""
    result = text
    if (not isinstance(result, types.StringTypes)):
        result = "%s" % text
    debug.trace_fmtd(8, "to_string({t}) => {r}", t=text, r=result)
    return result


def non_empty_file(filename):
    """Whether file exists and is non-empty"""
    non_empty = False
    try:
        non_empty = (os.path.getsize(filename) > 0)
    except OSError:
        debug.trace_fmtd(6, "Exception in non_empty_file: {exc}", exc=sys.exc_info())
    debug.trace_fmtd(5, "non_empty_file({f}) => {r}", f=filename, r=non_empty)
    return non_empty


def get_module_version(module_name):
    """Get version number for MODULE_NAME (string)"""
    # note: used in bash function (alias):
    #     python-module-version() = { python -c "print(get_module_version('$1))"; }'

    # Try to load the module with given name
    # TODO: eliminate eval and just import directly
    try:
        eval("import {m}".format(m=module_name))
    except:
        debug.trace_fmtd(6, "Exception importing module '{m}': {exc}",
                         m=module_name, exc=sys.exc_info())
        return "-1.-1.-1"

    # Try to get the version number for the module
    # TODO: eliminate eval and use attr()
    # TODO: try other conventions besides module.__version__ member variable
    version = "?.?.?"
    try:
        version = eval("module_name.__version__")
    except:
        debug.trace_fmtd(6, "Exception evaluating '{m}.__version__': {exc}",
                         m=module_name, exc=sys.exc_info())
        ## TODO: version = "0.0.0"
    return version

def intersection(list1, list2):
    """Return intersection of LIST1 and LIST2"""
    # note: wrapper around set.intersection used for tracing
    result = set(list1).intersection(set(list2))
    debug.trace_fmtd(7, "intersection({l1}, {l2}) => {r}",
                     l1=list1, l2=list2, r=result)
    return result

def difference(list1, list2):
    """Return set difference from LIST1 vs LIST2, preserving order"""
    # TODO: optmize (e.g., via a hash table)
    # EX: difference([5, 4, 3, 2, 1], [1, 2, 3]) => [5, 4]
    diff = []
    for item1 in list1:
        if item1 not in list2:
            diff.append(item1)
    debug.trace_fmtd(7, "difference({l1}, {l2}) => {d}",
                     l1=list1, l2=list2, d=diff)
    return diff

def to_float(text, default_value=0):
    """Interpret TEXT as integer, using default_value"""
    result = default_value
    try:
        result = float(text)
    except (TypeError, ValueError):
        debug.trace_fmtd(6, "Exception in to_int: {exc}", exc=sys.exc_info())
    return result

def to_int(text, default_value=0):
    """Interpret TEXT as integer, using default_value"""
    # TODO: use generic to_num with argument specifying type
    result = default_value
    try:
        result = int(text)
    except (TypeError, ValueError):
        debug.trace_fmtd(6, "Exception in to_int: {exc}", exc=sys.exc_info())
    return result

def to_bool(value):
    """Converts VALUE to boolean value, False iff in {0, False, and "False"}, ignoring case."""
    # TODO: add "off" as well
    value_text = str(value)
    bool_value = True
    if (value_text.lower() == "false") or (value_text == "0"):
        bool_value = False
    debug.trace_fmtd(7, "to_bool({v}) => {r}", v=value, r=bool_value)
    return bool_value


PRECISION = getenv_int("PRECISION", 6)
#
def round_num(value, precision=PRECISION):
    """Round VALUE [to PRECISION places, {p} by default]""".format(p=PRECISION)
    rounded_value = round(value, precision)
    debug.trace_fmtd(8, "round_num({v}, {p}) => {r}",
                     v=value, p=precision, r=rounded_value)
    return rounded_value

#-------------------------------------------------------------------------------


def main(args):
    """Supporting code for command-line processing"""
    debug.trace_fmtd(6, "main({a})", a=args)
    user = getenv_text("USER")
    print_stderr("Warning, {u}: Not intended for direct invocation".format(u=user))
    return

if __name__ == '__main__':
    main(sys.argv)
