#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Shows unicode information for characters in input. For each character, the output will
# consist of a row with the offset, printabe character, UTF-8 encoding, Unicode codepoint
# (i.e., U+XXXX), character category, and character name
#
# This also serves to illustrate the Unicode processing required under Python 2 vs. Python 3.
# For example, prior to processing the line, Python 2 requires using the UTF-8 encoder 
# (e.g., via 'unicode(line, "utf-8")' or via 'codecs.decode(line, "utf-8")'). In addition,
# prior to output, the line must be converted back into UTF-8 (see implementation of
# debug_print in tpo_common.py).
#
# Sample input:
#    <BOM>العربية (Arabic)<Newline>
#
#    see http://en.wikipedia.org/wiki/Arabic_language
#
# Sample output:
#
#   Offset Char	UTF-8	Unicode	Category Name
#   0		EFBBBF	U+feff	Cf	 ZERO WIDTH NO-BREAK SPACE
#   3	   ا	D8A7	U+0627	Lo	 ARABIC LETTER ALEF
#   5	   ل	D984	U+0644	Lo	 ARABIC LETTER LAM
#   7	   ع	D8B9	U+0639	Lo	 ARABIC LETTER AIN
#   9	   ر	D8B1	U+0631	Lo	 ARABIC LETTER REH
#   11	   ب	D8A8	U+0628	Lo	 ARABIC LETTER BEH
#   13	   ي	D98A	U+064a	Lo	 ARABIC LETTER YEH
#   15	   ة	D8A9	U+0629	Lo	 ARABIC LETTER TEH MARBUTA
#   17	    	20	U+0020	Zs	 SPACE
#   18	   (	28	U+0028	Ps	 LEFT PARENTHESIS
#   19	   A	41	U+0041	Lu	 LATIN CAPITAL LETTER A
#   20	   r	72	U+0072	Ll	 LATIN SMALL LETTER R
#   21	   a	61	U+0061	Ll	 LATIN SMALL LETTER A
#   22	   b	62	U+0062	Ll	 LATIN SMALL LETTER B
#   23	   i	69	U+0069	Ll	 LATIN SMALL LETTER I
#   24	   c	63	U+0063	Ll	 LATIN SMALL LETTER C
#   25	   )	29	U+0029	Pe	 RIGHT PARENTHESIS
#   26		0A	U+000a	Cc	 n/a
#
# Note:
# - U+FEFF serves dual purpose as byte order mark (BOM) and non-breaking space.
# - Based on sample from https://docs.python.org/2/howto/unicode.html.
#
# TODO:
# - See why print handles Unicode output but not [file.]write (used by debug_print):
#   http://stackoverflow.com/questions/8016236/python-unicode-handling-differences-between-print-and-sys-stdout-write
#

import sys
import unicodedata
import tpo_common as tpo

VERBOSE = tpo.getenv_boolean("VERBOSE", False, "Verbose output mode")

# Print information line by line
line_num = 0
offset = 0
for line in sys.stdin:
    line_num += 1
    tpo.debug_print("L%d: %s" % (line_num, line), 4)
    tpo.debug_print("type(line): %s" % type(line), 5)
    if VERBOSE:
        print(";; len={l}; text={t}".format(l=len(line), t=line))

    # Convert to unicode if necessary (e.g., python 2.x)
    if (sys.version_info[0] < 3):
        if (not isinstance(line, unicode)):
            tpo.debug_print("converting into unicode", 5)
            try:
                line = unicode(line, 'utf-8')
            except:
                tpo.print_stderr("Error: Exception during unicode conversion: " + str(sys.exc_info()))
            tpo.debug_print("type(line): %s" % type(line), 5)

    # Print description character by character
    print("\t".join(["Offset", "Char", "UTF-8", "Unicode", "Category", "Name"]))
    for i, unicode_char in enumerate(line):
        printable_char = (unicode_char if ord(unicode_char) >= 32 else ".")       
        tpo.debug_print("c%d: %s (%d)" % (i, printable_char, ord(unicode_char)), 6)

        # Get the unicode category and name for the character
        cat = "n/a"
        name = "n/a"
        try:
            cat = unicodedata.category(unicode_char)
            name = unicodedata.name(unicode_char)
        except:
            tpo.debug_print("Warning: Exception during unicodedata lookup: " + str(sys.exc_info()), 7)

        # Display the character info
        # note: under python 2.x, the output must be encoded at UTF-8
        utf8_bytes = unicode_char.encode('utf-8')
        if (sys.version_info[0] < 3) and (ord(unicode_char) >= 32):
            printable_char = utf8_bytes
        if (sys.version_info[0] < 3):
            utf8_codes = [ord(b) for b in utf8_bytes]
        else:
            utf8_codes = [b for b in utf8_bytes]
        encoding = "".join([("%02X" % b) for b in utf8_codes])
        print("\t".join([str(offset), printable_char, encoding, 'U+%04x' % ord(unicode_char), cat, name]))
        offset += len(utf8_bytes)
    print("")
