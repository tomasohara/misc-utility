#! /usr/bin/env python
#
# Add tracing to Javascript functions via console.debug. This scans the
# input for function definitions, and adds a console.debug trace call
# outputting the function name and parameter values.
#
# Sample input:
#   function setBlockText(msg) {
#        myBlock.textContent = msg;
#   }
#
#   function rescale_selector_image(index, element) {
#        rescale_image($(this));
#   }
#
# Sample output:
#   function set_block_text(msg) {
#        console.debug("set_block_text:" + " msg=" + msg);
#        my_block.textContent = msg;
#   }
#
#   function rescale_selector_image(index, element) {
#        console.debug("rescale_selector_image:" + " index=" + index + " element=" + element);
#        rescale_image($(this));
#   }
#
#-------------------------------------------------------------------------------
# TODO:
# - Add support for JSON.stringify.
# - Handle function arguments split across lines.
#
#

"""Adds function trace statements to Javascript code"""

import re
import sys

import debug

def main():
    """Entry point for script"""
    for i, line in enumerate(sys.stdin):
        sys.stdout.write(line)

        # Check for function definition and convert into trace statement
        trace_statement = ""
        match = re.search(r"^\s*function\s*(\w+)\s*\(([^{};]+)\)", line)
        if match:
            debug.trace_fmtd(4, "Found function definition at line {l}",
                             l=(i + 1))
            function_name = match.group(1)
            argument_spec = match.group(2)

            # Start trace: '    console.debug("<function>:"'
            trace_statement = "    console.debug('" + function_name + ":'"
            for argument in re.split(", *", argument_spec):

                # Add in argument:   ' "<argument>=" + <argument>'
                trace_statement += " + ' " + argument + "='" + " + " + argument 

            # Add indicator if no arguments (e.g., " n/a")
            if trace_statement.endswith(":"):
                trace_statement += " 'n/a'"
                
            # Close trace and output
            trace_statement += ");\n"
            sys.stdout.write(trace_statement)

                
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
