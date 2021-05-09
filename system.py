#! /usr/bin/env python
#
# Functions for system related access, such as running command or
# getting environment values.
#
# TODO:
# - Finish python 2/3 special case handling via six package (e.g., in order
#   to remove stupud pylint warnings).
# - Rename as system_utils so clear that non-standard package.
# - Add support for maintaining table of environment variables with optional descriptions.
# - Add safe_int from tpo_common, as includes support for base (e.g., 16 for hex).
# - Reconcile against functions in tpo_common.py (e.g., to make sure functionality covered and similar tracing supported).
#

"""System-related functions"""

## OLD: if sys.version_info.major == 2:
## OLD:     from __future__ import print_function
from __future__ import print_function

# Standard packages
from collections import defaultdict, OrderedDict
import datetime
import inspect
import os
import pickle
import re
import six
import sys
import time
import urllib

# Local packages
import debug
from debug import UTF8

# Constants
STRING_TYPES = six.string_types
MAX_SIZE = six.MAXSIZE
MAX_INT = MAX_SIZE

#-------------------------------------------------------------------------------
# Support for needless python changes

def maxint():
    """Maximum size for an integer"""
    # Note: this is just a sanity check
    debug.assertion(MAX_INT == MAX_SIZE)
    return MAX_INT

#-------------------------------------------------------------------------------
# Support for environment variable access
# TODO: Put in separate module

env_options = {}
env_defaults = {}
#
def register_env_option(var, description, default):
    """Register environment VAR as option with DESCRIPTION and DEFAULT"""
    debug.trace_fmt(7, "register_env_option({v}, {d})", v=var, d=description)
    global env_options
    global env_defaults
    env_options[var] = (description or "")
    env_defaults[var] = default
    return


def get_registered_env_options():
    """Returns list of environment options registered via register_env_option"""
    option_names = [k for k in env_options if (env_options[k] and env_options[k].strip())]
    debug.trace_fmt(5, "get_registered_env_options() => {ON}", on=option_names)
    return option_names


def get_environment_option_descriptions(include_all=None, include_default=None, indent=" "):
    """Returns list of environment options and their descriptions"""
    debug.trace_fmt(5, "env_options={eo}", eo=env_options)
    debug.trace_fmt(5, "env_defaults={ed}", ed=env_defaults)
    if include_all is None:
        include_all = debug.verbose_debugging()
    if include_default is None:
        include_default = True
    #
    def _format_env_option(opt):
        """Returns OPT description and optionally default value (if INCLUDE_DEFAULT)"""
        debug.trace_fmt(7, "_format_env_option({o})", o=opt)
        desc_spec = env_options.get(opt, "_")
        default_spec = ""
        if include_default:
            default_value = env_defaults.get(opt, None)
            has_default = (default_value is not None)
            default_spec = ("(%s)" % default_value) if has_default else "n/a"
        default_spec = default_spec.replace("\n", "\\n")
        return (opt, desc_spec + indent + default_spec)
    #
    option_descriptions = [_format_env_option(opt) for opt in env_options if (env_options[opt] or include_all)]
    debug.trace_fmt(5, "get_environment_option_descriptions() => {od}",
                    od=option_descriptions)
    return option_descriptions


def formatted_environment_option_descriptions(sort=False, include_all=None, indent="\t"):
    """Returns string list of environment options and their descriptions (separated by newlines and tabs), optionally SORTED"""
    option_info = get_environment_option_descriptions(include_all)
    if sort:
        option_info = sorted(option_info)
    entry_separator = "\n%s" % indent
    descriptions = entry_separator.join(["%s%s%s" % (opt, indent, (desc if desc else "n/a")) for (opt, desc) in option_info])
    debug.trace_fmt(6, "formatted_environment_option_descriptions() => {d}",
                    d=descriptions)
    return descriptions


def getenv_text(var, default="", description=None, helper=False):
    """Returns textual value for environment variable VAR (or DEFAULT value).
    Note: HELPER indicates that this call is in support of another getenv-type function (e.g., getenv_bool), so that tracing is only shown at higher verbosity level (e.g., 6 not 5)."""
    register_env_option(var, description, default)
    text_value = os.getenv(var)
    if not text_value:
        debug.trace_fmtd(6, "getenv_text: no value for var {v}", v=var)
        text_value = default
    trace_level = 6 if helper else 5
    debug.trace_fmtd(trace_level, "getenv_text('{v}', [def={dft}], [desc={desc}], [helper={hlpr}]) => {r}",
                     v=var, dft=default, desc=description, hlpr=helper, r=text_value)
    return (text_value)


DEFAULT_GETENV_BOOL = False
#
def getenv_bool(var, default=DEFAULT_GETENV_BOOL, _description=None):
    """Returns boolean flag based on environment VAR (or DEFAULT value)"""
    # Note: "0" or "False" is interpreted as False, and any other value as True.
    bool_value = default
    value_text = getenv_text(var, helper=True)
    if value_text.strip():
        bool_value = to_bool(value_text)
    debug.trace_fmtd(5, "getenv_bool({v}, {d}) => {r}",
                     v=var, d=default, r=bool_value)
    return (bool_value)
#
getenv_boolean = getenv_bool


def getenv_number(var, default=-1.0, _description=None, helper=False):
    """Returns number based on environment VAR (or DEFAULT value)."""
    # Note: use getenv_int or getenv_float for typed variants
    num_value = default
    value_text = getenv_text(var, helper=True)
    if value_text.strip():
         num_value = to_float(value_text)
    trace_level = 6 if helper else 5
    debug.trace_fmtd(trace_level, "getenv_number({v}, {d}) => {r}",
                     v=var, d=default, r=num_value)
    return (num_value)


def getenv_int(var, default=-1, _description=None):
    """Version of getenv_number for integers"""
    value = getenv_number(var, default=default, helper=True)
    if (value is not None):
        value = to_int(value)
    debug.trace_fmtd(5, "getenv_int({v}, {d}) => {r}",
                     v=var, d=default, r=value)
    return (value)
#
getenv_integer = getenv_int


def getenv_float(var, default=-1, _description=None):
    """Version of getenv_number for floating point values"""
    result = getenv_number(var, default=default, helper=True)
    debug.trace_fmtd(5, "getenv_float({v}, {d}) => {r}",
                     v=var, d=default, r=result)
    return (result)

#-------------------------------------------------------------------------------
# Miscellaneous functions

def get_exception():
    """Return information about the exception that just occurred"""
    # Note: Convenience wrapper to avoid need to import sys in simple scripts.
    return sys.exc_info()

def print_stderr(text, **kwargs):
    """Output TEXT to standard error"""
    # TODO: weed out calls that use (text.format(...)) rather than (text, ...)
    formatted_text = text
    try:
        # Note: to avoid interpolated text as being interpreted as variable
        # references, this function should do the formatting
        # ex: print_stderr("hey {you}".format(you="{u}")) => print_stderr("hey {you}".format(you="{u}"))
        debug.assertion(kwargs or (not re.search(r"{\S*}", text)))
        formatted_text = text.format(**kwargs)
    except(KeyError, ValueError, UnicodeEncodeError):
        sys.stderr.write("Warning: Problem in print_stderr: {exc}\n".format(
            exc=get_exception()))
        if debug.verbose_debugging():
            print_full_stack()
    print(formatted_text, file=sys.stderr)
    return


def setenv(var, value):
    """Set environment VAR to VALUE"""
    debug.trace_fmtd(5, "setenv({v}, {val})", v=var, val=value)
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
        debug.trace_fmtd(3, "Unable to produce stack trace: {exc}", exc=get_exception())
    stream.write("\n")
    return


def get_current_function_name():
    """Returns name of current function that is running"""
    function_name = "???"
    try:
        current_frame = inspect.stack()[1]
        function_name = current_frame[3]
    except:
        debug.trace_fmtd(3, "Unable to resolve function name: {exc}", exc=get_exception())
    return function_name


def save_object(file_name, obj):
    """Saves OBJ to FILE_NAME in pickle format"""
    # Note: The data file is created in binary mode to avoid quirk under Windows.
    # See https://stackoverflow.com/questions/556269/importerror-no-module-named-copy-reg-pickle.
    debug.trace_fmtd(6, "save_object({f}, _)", f=file_name)
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f)
    except (AttributeError, IOError, TypeError, ValueError):
        debug.trace_fmtd(1, "Error: Unable to save object to {f}: {exc}",
                         f=file_name, exc=get_exception())
    return

    
def load_object(file_name, ignore_error=False):
    """Loads object from FILE_NAME in pickle format"""
    # Note: Reads in binary mode to avoid unicode decode error. See
    #    https://stackoverflow.com/questions/32957708/python-pickle-error-unicodedecodeerror
    obj = None
    try:
        with open(file_name, 'rb') as f:
            obj = pickle.load(f)
    except (AttributeError, IOError, TypeError, ValueError):
        if (not ignore_error):
            print_stderr("Error: Unable to load object from {f}: {exc}".
                         format(f=file_name, exc=get_exception()))
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
    try:
        if not re.search("%[0-9A-F]{2}", text):
            if sys.version_info.major > 2:
                result = urllib.parse.quote_plus(text)      # pylint: disable=no-member
            else:
                result = urllib.quote_plus(to_utf8(text))   # pylint: disable=no-member
    except (TypeError, ValueError):
        debug.trace_fmtd(6, "Exception quoting url text 't': {exc}",
                         t=text, exc=get_exception())
        
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
    if (sys.version_info.major > 2):
        # TODO: move import to top
        import html                    # pylint: disable=import-outside-toplevel, import-error
        result = html.escape(text)
    else:
        import cgi                     # pylint: disable=import-outside-toplevel, import-error
        result = cgi.escape(text, quote=True)    # pylint: disable=deprecated-method
    debug.trace_fmtd(6, "out escape_html_text({t}) => {r}", t=text, r=result)
    return result


def unescape_html_text(text):
    """Remove entity encoding, etc. from TEXT (i.e., undo"""
    # Note: This is wrapper around html.unescape (Python 3+) or
    # HTMLParser.unescape (Python 2).
    # See https://stackoverflow.com/questions/21342549/unescaping-html-with-special-characters-in-python-2-7-3-raspberry-pi.
    # EX: unescape_html_text(escape_html_text("<2/")) => "<2/"
    debug.trace_fmtd(7, "in unescape_html_text({t})", t=text)
    result = ""
    if (sys.version_info.major > 2):
        # TODO: see if six.py supports html-vs-cgi:unescape
        import html                   # pylint: disable=import-outside-toplevel, import-error
        result = html.unescape(text)
    else:
        import HTMLParser             # pylint: disable=import-outside-toplevel, import-error
        html_parser = HTMLParser.HTMLParser()
        result = html_parser.unescape(text)
    debug.trace_fmtd(6, "out unescape_html_text({t}) => {r}", t=text, r=result)
    return result


NEWLINE = "\n"
TAB = "\t"
#
class stdin_reader(object):
    """Iterator for reading from stdin that replaces runs of whitespace by tabs"""
    # TODO: generalize to file-based iterator
    
    def __init__(self, *args, **kwargs):
        """Class constructor"""
        debug.trace_fmtd(5, "Script.__init__({a}): keywords={kw}; self={s}",
                         a=",".join(args), kw=kwargs, s=self)
        self.delimiter = kwargs.get('delimiter', "\t")
        super(stdin_reader, self).__init__(*args, **kwargs)
    
    def __iter__(self):
        """Returns first line in stdin iteration (empty string upon EOF)"""
        return self.__next__()

    def __next__(self):
        """Returns next line in stdin iteration (empty string upon EOF)"""
        try:
            line = self.normalize_line(input())
        except EOFError:
            line = ""
        return line

    def normalize_line(self, original_line):
        """Normalize line (e.g., replacing spaces with single tab)"""
        debug.trace_fmtd(6, "in normalize_line({ol})", ol=original_line)
        line = original_line
        # Remove trailing newline
        if line.endswith(NEWLINE):
            line = line[:-1]
        # Replace runs of spaces with a single tab
        if (self.delimiter == TAB):
            line = re.sub(r"  *", TAB, line)
        # Trace the revised line and return it
        debug.trace_fmtd(6, "normalize_line() => {l}", l=line)
        return line


def read_entire_file(filename):
    """Read all of FILENAME and return as a string"""
    data = ""
    try:
        with open(filename) as f:
            data = from_utf8(f.read())
    except IOError:
        debug.trace_fmtd(1, "Error: Unable to read file '{f}': {exc}",
                         f=filename, exc=get_exception())
    debug.trace_fmtd(7, "read_entire_file({f}) => {r}", f=filename, r=data)
    return data
#
read_file = read_entire_file


def read_lookup_table(filename, skip_header=False):
    """Reads FILENAME and returns as hash lookup"""
    hash_table = {}
    line_num = 0
    try:
        with open(filename) as f:
            for line in f:
                line_num += 1
                if (skip_header and (line_num == 1)):
                    continue
                line = from_utf8(line)
                # TODO: trap exception and lowercase key
                if "\t" in line:
                    (key, value) = line.split("\t", 1)
                    hash_table[key] = value
                else:
                    debug.trace(2, "Ignoring line w/o tab {n}: {l}", n=line_num, l=line)
    except (IOError, ValueError):
        debug.trace_fmtd(1, "Error creating lookup from '{f}': {exc}",
                         f=filename, exc=get_exception())
    debug.trace_fmtd(7, "read_lookup_table({f}) => {r}", f=filename, r=hash_table)
    return hash_table


def create_boolean_lookup_table(filename):
    """Create lookup hash table from string keys to boolean occurrence indicator. Notes:
    - The keys are made lowercase.
    - The hash is of type defaultdict(bool).
    """
    # TODO: allow for tab-delimited value to be ignored
    # TODO: make the result hash case insensitive (or add accessor such as check_boolean_lookup_table)
    debug.trace_fmt(4, "create_boolean_lookup_table({f})", f=filename)
    lookup_hash = defaultdict(bool)
    try:
        with open(filename) as f:
            for line in f:
                key = line.strip().lower()
                debug.assertion("\t" not in key)
                lookup_hash[key] = True
    except (IOError, ValueError):
        debug.trace_fmtd(1, "Error: Creating boolean lookup from '{f}': {exc}",
                         f=filename, exc=get_exception())
    debug.trace_fmt(7, "create_boolean_lookup_table => {h}", h=lookup_hash)
    return lookup_hash


def write_file(filename, text):
    """Create FILENAME with TEXT"""
    debug.trace_fmt(7, "write_file({f}, {t})", f=filename, t=text)
    try:
        if not isinstance(text, STRING_TYPES):
            text = to_string(text)
        with open(filename, "w") as f:
            ## OLD: f.write(to_utf8(text) + "\n")
            f.write(to_utf8(text))
            if not text.endswith("\n"):
                f.write("\n")
    except (IOError, ValueError):
        debug.trace_fmtd(1, "Error: Problem writing file '{f}': {exc}",
                         f=filename, exc=get_exception())
    return


def get_file_modification_time(filename, as_float=False):
    """Get the time the FILENAME was last modified, optional AS_FLOAT (instead of default string).
    Note: Returns None if file doesn't exist."""
    # TODO: document how the floating point version is interpretted
    # See https://stackoverflow.com/questions/237079/how-to-get-file-creation-modification-date-times-in-python
    mod_time = None
    if file_exists(filename):
        mod_time = os.path.getmtime(filename)
        if not as_float:
            mod_time = str(datetime.datetime.fromtimestamp(mod_time))
    debug.trace_fmtd(5, "get_file_modification_time({f}) => {t}", f=filename, t=mod_time)
    return mod_time


def remove_extension(filename):
    """Return FILENAME without final extension"""
    # ex: remove_extension("document.pdf") => "document")
    new_filename = re.sub(r"\.[^\.]*$", "", filename)
    debug.trace_fmtd(4, "remove_extension({f}) => {r}", f=filename, r=new_filename)
    return new_filename


def file_exists(filename):
    """Returns True iff FILENAME exists"""
    does_exist = os.path.exists(filename)
    debug.trace_fmtd(5, "file_exists({f}) => {r}", f=filename, r=does_exist)
    return does_exist


def get_file_size(filename):
    """Returns size of FILENAME or -1 if not found"""
    size = -1
    if file_exists(filename):
        size = os.path.getsize(filename)
    debug.trace_fmtd(5, "get_file_size({f}) => {s}", f=filename, s=size)
    return size


def form_path(*filenames):
    """Wrapper around os.path.join over FILENAMEs (with tracing)"""
    path = os.path.join(*filenames)
    debug.trace_fmt(6, "form_path{f} => {p}", f=tuple(filenames), p=path)
    return path


def is_directory(path):
    """Determins wther PATH represents a directory"""
    is_dir = os.path.isdir(path)
    debug.trace_fmt(6, "is_dir{p} => {r}", p=path, r=is_dir)
    return is_dir


def create_directory(path):
    """Wrapper around os.mkdir over PATH (with tracing)"""
    debug.trace_fmt(7, "create_directory({p})", p=path)
    if not os.path.exists(path):
        os.mkdir(path)
        debug.trace_fmt(6, "os.mkdir({p})", p=path)
    else:
        debug.assertion(os.path.isdir(path))
    return


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
                local_filename, headers = urllib.request.urlretrieve(url, filename)      # pylint: disable=no-member
            else:
                local_filename, headers = urllib.urlretrieve(url, filename)      # pylint: disable=no-member
            debug.trace_fmtd(5, "=> local file: {f}; headers={{h}}",
                             f=local_filename, h=headers)
        except IOError:
            debug.raise_exception(6)
            debug.trace_fmtd(1, "Error: Unable to download {u}: {exc}",
                             u=url, exc=get_exception())
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
    if ((sys.version_info.major < 3) and (isinstance(text, unicode))):   # pylint: disable=undefined-variable
        result = result.encode(UTF8, 'ignore')
    debug.trace_fmtd(8, "to_utf8({t}) => {r}", t=text, r=result)
    return result


def from_utf8(text):
    """Convert TEXT to Unicode from UTF-8"""
    result = text
    if ((sys.version_info.major < 3) and (not isinstance(result, unicode))):     # pylint: disable=undefined-variable
        result = result.decode(UTF8, 'ignore')
    debug.trace_fmtd(8, "from_utf8({t}) => {r}", t=text, r=result)
    return result


def to_unicode(text, encoding=None):
    """Ensure TEXT in ENCODING is Unicode, such as from the default UTF8"""
    # TODO: rework from_utf8 in terms of this
    result = text
    if ((sys.version_info.major < 3) and (not isinstance(result, unicode))):     # pylint: disable=undefined-variable
        if not encoding:
            encoding = UTF8
        result = result.decode(encoding, 'ignore')
    debug.trace_fmtd(8, "to_unicode({t}, [{e}]) => {r}", t=text, e=encoding, r=result)
    return result


def from_unicode(text, encoding=None):
    """Convert TEXT to ENCODING from Unicode, such as to the default UTF8"""
    # TODO: rework to_utf8 in terms of this
    result = text
    if ((sys.version_info.major < 3) and (isinstance(text, unicode))):    # pylint: disable=undefined-variable
        if not encoding:
            encoding = UTF8
        result = result.encode(encoding, 'ignore')
    debug.trace_fmtd(8, "from_unicode({t}, [{e}]) => {r}", t=text, e=encoding, r=result)
    return result


def to_string(text):
    """Ensure TEXT is a string type"""
    # Notes: Uses string formatting operator % for proper Unicode handling.
    # Gotta hate Python: doubly stupid changes from version 2 to 3 required special case handling: Unicode type dropped and bytes not automatically treated as string!
    result = text
    if isinstance(result, bytes):
        result = result.decode(UTF8)
    if (not isinstance(result, STRING_TYPES)):
        result = "%s" % text
    debug.trace_fmtd(8, "to_string({t}) => {r}", t=text, r=result)
    return result


def chomp(text, line_separator=os.linesep):
    """Removes trailing occurrence of LINE_SEPARATOR from TEXT"""
    # EX: chomp("abc\n") => "abc"
    # EX: chomp("http://localhost/", "/") => "http://localhost"
    result = text
    if result.endswith(line_separator):
        new_len = len(result) - len(line_separator)
        result = result[:new_len]
    debug.trace_fmt(8, "chomp({t}, {sep}) => {r}", 
                    t=text, sep=line_separator, r=result)
    return result


def non_empty_file(filename):
    """Whether file exists and is non-empty"""
    non_empty = (file_exists(filename) and (os.path.getsize(filename) > 0))
    debug.trace_fmtd(5, "non_empty_file({f}) => {r}", f=filename, r=non_empty)
    return non_empty


def get_module_version(module_name):
    """Get version number for MODULE_NAME (string)"""
    # note: used in bash function (alias):
    #     python-module-version() = { python -c "print(get_module_version('$1))"; }'

    # Try to load the module with given name
    # TODO: eliminate eval and just import directly
    try:
        eval("import {m}".format(m=module_name))             # pylint: disable=eval-used
    except:
        debug.trace_fmtd(6, "Exception importing module '{m}': {exc}",
                         m=module_name, exc=get_exception())
        return "-1.-1.-1"

    # Try to get the version number for the module
    # TODO: eliminate eval and use getattr()
    # TODO: try other conventions besides module.__version__ member variable
    version = "?.?.?"
    try:
        version = eval("module_name.__version__")            # pylint: disable=eval-used
    except:
        debug.trace_fmtd(6, "Exception evaluating '{m}.__version__': {exc}",
                         m=module_name, exc=get_exception())
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


def append_new(in_list, item):
    """Returns copy of LIST with ITEM included unless already in it"""
    # ex: append_new([1, 2], 3]) => [1, 2, 3]
    # ex: append_new([1, 2, 3], 3]) => [1, 2, 3]
    result = in_list[:]
    if item not in result:
        result.append(item)
    debug.trace_fmt(7, "append_new({l}, {i}) => {r}",
                    l=in_list, i=item, r=result)
    return result


def just_one_true(in_list):
    """True iff only one element of IN_LIST is considered True"""
    # TODO: Trap exceptions (e.g., string input)
    is_true = (1 == sum([int(b) for b in in_list]))         # pylint: disable=misplaced-comparison-constant
    debug.trace_fmt(6, "just_one_true({l}) => {r}", l=in_list, r=is_true)
    return is_true


def unique_items(in_list, prune_empty=False):
    """Returns unique items from IN_LIST, preserving order and optionally PRUN[ing]_EMPTY items"""
    # EX: unique_items([1, 2, 3, 2, 1]) => [1, 2, 3]
    ordered_hash = OrderedDict()
    for item in in_list:
        if item or (not prune_empty):
            ordered_hash[item] = True
    result = ordered_hash.keys()
    debug.trace_fmt(8, "unique_items({l}) => {r}", l=in_list, r=result)
    return result


def to_float(text, default_value=0.0):
    """Interpret TEXT as float, using DEFAULT_VALUE"""
    result = default_value
    try:
        result = float(text)
    except (TypeError, ValueError):
        debug.trace_fmtd(6, "Exception in to_float: {exc}", exc=get_exception())
    return result
#
safe_float = to_float


def to_int(text, default_value=0):
    """Interpret TEXT as integer, using DEFAULT_VALUE"""
    # TODO: use generic to_num with argument specifying type
    result = default_value
    try:
        result = int(text)
    except (TypeError, ValueError):
        debug.trace_fmtd(6, "Exception in to_int: {exc}", exc=get_exception())
    return result
#
safe_int = to_int


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
    debug.trace_fmtd(8, "round_num({v}, [prec={p}]) => {r}",
                     v=value, p=precision, r=rounded_value)
    return rounded_value


def sleep(num_seconds, trace_level=5):
    """Sleep for NUM_SECONDS"""
    debug.trace_fmtd(trace_level, "sleep({ns}, [tl={tl}])",
                     ns=num_seconds, tl=trace_level)
    time.sleep(num_seconds)
    return

#-------------------------------------------------------------------------------
# Command line usage

def main(args):
    """Supporting code for command-line processing"""
    debug.trace_fmtd(6, "main({a})", a=args)
    user = getenv_text("USER")
    print_stderr("Warning, {u}: Not intended for direct invocation".format(u=user))
    debug.trace_fmt(4, "FYI: maximum integer is {maxi}", maxi=maxint())
    return

if __name__ == '__main__':
    main(sys.argv)
