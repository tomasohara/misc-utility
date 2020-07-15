#! /usr/bin/env python
#
# spell.py: simple spell checker (e.g., via Enchant module)
#

import sys
import fileinput
import re
import enchant			# spell checking
from tpo_common import debug_print

# Process command line
i = 1
show_usage = (i == len(sys.argv))
while (i < len(sys.argv)) and (sys.argv[i][0] == "-"):
    if (sys.argv[i] == "--help"):
        show_usage = True
    elif (sys.argv[i] == "-"):
        pass
    else:
        print("Error: unexpected argument '%s'" % sys.argv[i])
        show_usage = True
    i += 1
if (show_usage):
    print("Usage: %s [options] input-file" % sys.argv[0])
    print("")
    print("Options: [--help]")
    print("")
    print("Example: %s qiqci-query-keywords.list" % sys.argv[0])
    sys.exit()
# Discard any used arguments (for sake of fileinput)
if (i > 1):
    sys.argv = [sys.argv[0]] + sys.argv[i:]

# Initialize spell checking
speller = enchant.Dict("en_US")

# Check input
for line in fileinput.input():
    line = line.strip()
    debug_print("L%d: %s" % (fileinput.filelineno(), line), 5)

    # Extract word tokens and print those not recognized 
    word_tokens = [t.strip() for t in re.split("\W+", line.lower(), re.LOCALE|re.UNICODE) if (len(t.strip()) > 0)]
    debug_print("tokens: %s" % word_tokens, 4)
    for w in word_tokens:
        if not speller.check(w):
            print(w)
