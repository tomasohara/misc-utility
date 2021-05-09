#! /usr/bin/env python
#
# unicode-converter.py: converts between various file formats (e.g., Windows-1259 to UTF-8)
#
# Notes:
# - https://stackoverflow.com/questions/31207287/converting-utf-16-to-utf-8.
# - https://stackoverflow.com/questions/492483/setting-the-correct-encoding-when-piping-stdout-in-python
#
# TODO:
# - Remove extraneous byte order marks (BOM's), such as after start of file.
#

"""Unicode converter (e.g., UTF-16 to UTF-8)"""

import argparse
import codecs
import os
import sys

import tpo_common as tpo
import debug
import system

UTF8 = "utf-8"
UTF16 = "utf-16"
TO_UTF8 = "to-utf8"
TO_UTF16 = "to-utf16"
FROM_UTF8 = "from-utf8"
FROM_UTF16 = "from-utf16"
TO = "to"
FROM = "from"


def option(label):
    """Convert LABEL into an option (e.g., for argparse)"""
    # ex: option("skip-headers") => "--skip-headers"
    return "--" + label


def arg(label):
    """Convert LABEL into parsed-args key"""
    # ex: arg("skip-headers") => "skip_headers"
    return label.replace("-", "_")

class EncodedStdout:
    """Context for shadowing sys.stdout with encoding-specific version"""
    # usage: with EncodedStdout('utf-8'):
    #            ...

    def __init__(self, enc):
        """Constructor: save current stdout"""
        self.enc = enc
        self.stdout = sys.stdout

    def __enter__(self):
        """Initialize context by replacing stdout"""
        if sys.stdout.encoding is None:
            w = codecs.getwriter(self.enc)
            sys.stdout = w(sys.stdout)

    def __exit__(self, exc_ty, exc_val, tb):
        """Exit context by restoring stdout"""
        sys.stdout = self.stdout

def main():
    """Entry point for script"""
    tpo.debug_print("main(): sys.argv=%s" % sys.argv, 4)

    # Check command-line arguments
    # TODO: add in detailed usage notes w/ environment option descriptions (see google_word2vec.py)
    parser = argparse.ArgumentParser(description="TODO: what the script does")
    # TODO: use capitalized script description but lowercase argument help
    parser.add_argument(option(FROM_UTF8), default=False, action='store_true',
                        help="Convert from UTF8")
    parser.add_argument(option(FROM_UTF16), default=False, action='store_true',
                        help="Convert from UTF-16")
    parser.add_argument(option(TO_UTF8), default=False, action='store_true',
                        help="Convert to UTF8")
    parser.add_argument(option(TO_UTF16), default=False, action='store_true',
                        help="Convert to UTF-16")
    parser.add_argument(option(FROM), default="",
                        help="Input encoding")
    parser.add_argument(option(TO), default="",
                        help="Output encoding")
    parser.add_argument("filename", nargs='?', default='-',
                        help="Input filename")
    args = vars(parser.parse_args())
    tpo.debug_print("args = %s" % args, 5)
    to_encoding = args['to']
    from_encoding = args['from']
    if not to_encoding and args[arg(TO_UTF8)]:
        to_encoding = UTF8
    if not to_encoding and args[arg(TO_UTF16)]:
        to_encoding = UTF16
    if not from_encoding and args[arg(FROM_UTF8)]:
        from_encoding = UTF8
    if not from_encoding and args[arg(FROM_UTF16)]:
        from_encoding = UTF16
    filename = args['filename']
    debug.trace_fmtd(4, "from={fe} to={te} filename={f}", fe=from_encoding, te=to_encoding, f=filename)

    # TODO: put supporting code in module to cut down on boilerplate code
    input_stream = sys.stdin
    if filename != "-":
        if not os.path.exists(filename):
            system.print_stderr("Unable to find {f}", f=filename)
            return
        ## input_stream = open(filename, "r")
        input_stream = codecs.open(filename, "rb", from_encoding)
        if not input_stream:
            system.print_stderr("Unable to open {f}", f=filename)
            return
    else:
        tpo.debug_print("Processing stdin", 6)
        input_stream.encoding = from_encoding
    ## BAD: sys.stdout.encoding = to_encoding
    ## sys.stdout = codecs.getwriter('utf8')(sys.stdout)

    # Process input line by line
    line_num = 0
    with EncodedStdout(to_encoding):
        for line in input_stream:
            ## line = line.decode(from_encoding, 'ignore')
            debug.trace_fmtd(6, "L{n}: {l}", n=line_num, l=line)
            line_num += 1
            ## sys.stdout.write(line.encode(to_encoding))
            ## sys.stdout.write(line.encode(to_encoding, 'ignore'))
            sys.stdout.write(line)

    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
