#! /usr/bin/env python
#
# Trains text categorizer using Scikit-Learn. See
#    http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
#
# TODO:
# - Rename to something like apply_text_categorizer.py.
#
# Copyright (c) 2012-2018 Thomas P. O'Hara
#

"""Trains text categorization"""

import sys

import debug
import system
from text_categorizer import TextCategorizer


SHOW_REPORT = system.getenv_bool("SHOW_REPORT", False)


def usage():
    """Show command-line usage"""
    # TODO: remove path from script filename
    script = (__file__ or "n/a")
    system.print_stderr("Usage: {scr} training-file model-file [testing]".format(scr=script))
    system.print_stderr("")
    system.print_stderr("Notes:")
    system.print_stderr("- Use - to indicate the file is not needed (e.g., existing training model).")
    system.print_stderr("- You need to supply either training file or model file.")
    system.print_stderr("- The testing file is optional when training.")
    return


def main(args=None):
    """Entry point for script"""
    debug.trace_fmtd(4, "main(): args={a}", a=args)

    # Check command line arguments
    if args is None:
        args = sys.argv
        if len(args) <= 2:
            usage()
            return
    training_filename = args[1]
    model_filename = args[2]
    testing_filename = None
    if (len(args) > 3):
        testing_filename = args[3]

    # Train text categorizer and save model to specified file
    text_cat = TextCategorizer()
    new_model = False
    accuracy = None
    if training_filename and (training_filename != "-"):
        text_cat.train(training_filename)
        new_model = True
    if model_filename and (model_filename != "-"):
        if new_model:
            text_cat.save(model_filename)
        else:
            text_cat.load(model_filename)
    if testing_filename and (testing_filename != "-"):
        accuracy = text_cat.test(testing_filename, report=SHOW_REPORT)
        print("Accuracy over {f}: {acc}".format(acc=accuracy, f=testing_filename))
    # Show usage if nothing done (e.g., due to too many -'s for filenames)
    if (not (new_model or accuracy)):
        usage()
              
    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
