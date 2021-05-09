#! /usr/bin/env python
#
# extract_wiki_token.py: Extracts tokens from each of the articles in a
# Wikipedia dump
#
#

"""Extraction wikipedia article tokens"""

import re
import sys

from gensim.corpora import WikiCorpus

import debug
import system

INCLUDE_TEMPLATES = system.getenv_bool("INCLUDE_TEMPLATES", False)
NORMALIZE_TITLE = system.getenv_bool("NORMALIZE_TITLE", False)

def main(args=None):
    """Entry point for script"""
    debug.trace_fmtd(4, "main(): args={a}", a=args)
    if args is None:
        args = sys.argv
        if len(args) <= 1:
            system.print_stderr("{f}:main: need to supply wiki dump filename".
                                format(f=(__file__ or "n/a")))
            return
    filename = args[1]

    # Open Wikipedia corpus but block unnecessary dictionary-building pass
    dummy_dict = {1: 'one'}
    wiki = WikiCorpus(filename, dictionary=dummy_dict)
    wiki.metadata = True

    # Print title and tokens in tabular format
    for (tokens, (_pageid, title)) in wiki.get_texts():
        ## BAD: output = "{title}\t{text}".format(title=title, text=(" ".join(tokens)))
        # Filter templates
        if (title.startswith("template:") and (not INCLUDE_TEMPLATES)):
            debug.trace_fmtd(4, "Ignoring template article {t}", t=title)
            continue
        # Make sure title uses underscores for spaces, omitting extraneous ones
        if NORMALIZE_TITLE:
            title = re.sub("(^ +)|( +$)", title, "")
            title = title.replace(" ", "_")
        # Output the article
        try:
            output = title + "\t" + (" ".join(tokens))
            print(system.to_utf8(output))
        except:
            system.print_stderr("Problem with " + str(_pageid))
    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
