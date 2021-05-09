#! /usr/bin/env python
#
# download-wiki-category.py: downloads all Wikipedia articles in category (and descendant cateories)
#
#

#------------------------------------------------------------------------
# Library packages

from common import *
## TODO: from common import getenv_boolean, getenv_text, print_stderr, debug_print, debug_timestamp

import sys
import re
import commands

from urllib import FancyURLopener
import time

#------------------------------------------------------------------------
# Globals

processed = dict()
base_url = "http://en.wikipedia.org"

OUTPUT_DIR = getenv_text("OUTPUT_DIR", ".")
SKIP_SUBCATS = getenv_boolean("SKIP_SUBCATS", False)
SKIP_PAGES = getenv_boolean("SKIP_PAGES", False)
MAKE_SUBDIRS = getenv_boolean("MAKE_SUBDIRS", False)
URL_SLEEP = getenv_integer("URL_SLEEP", 1)
PAGES_START = getenv_text("PAGES_START", "Pages in category")
PAGES_END = getenv_text("PAGES_END", "Media in category")

#------------------------------------------------------------------------
# Classes

class my_re:
    """Class implementing regex search that saves match results (for use in conditions)"""
    match = None

    @staticmethod
    def search(regex, text, flags=0):
        """Search for REGEX in TEXT with optional FLAGS"""
        my_re.match = re.search(regex, text, flags)
        if my_re.match:
            # TODO: while my_re.match.group(i): print my_re.match.group(i)
            debug_print("match: %s" % my_re.match.group(), 6)
        return (my_re.match)

    @staticmethod
    def get_match():
        """Return MatchObject for last search"""
        return (my_re.match)

class MyURLOpener(FancyURLopener):
    """URLOpener with personal version (for user-agent): see http://wolfprojects.altervista.org/changeua.php"""
    version = 'TPO/1.0 (tomasohara@gmail.com)'


#------------------------------------------------------------------------
# Functions

def make_directory(dir):
    """Creates file system DIR (ignoring errors)"""
    debug_print("make_directory(%s)" % dir, 4)
    try:
        os.mkdir(dir)
    except:
        print_stderr("Unable to create directory '%s': %s" % (dir, sys.exc_info()))
    return

def change_directory(dir):
    """Changes into file system DIR (ignoring errors)"""
    debug_print("change_directory(%s)" % dir, 4)
    try:
        os.chdir(dir)
    except:
        print_stderr("Unable to change into directory '%s': %s" % (filename, sys.exc_info()))
    return

def run_command(command_line, level=5):
    """Runs COMMAND_LINE and returns the output (as a string), with debug tracing at specified LEVEL"""
    # Issue command
    debug_print("Running command: %s" % command_line, level=level)
    (status, output) = commands.getstatusoutput(command_line)
    if (status != 0):
        print_stderr("Warning: problem running command (status=%d): %s" % (status, command_line))

    # Return output
    debug_print("Command output: %s\n" % output, level=level+1)
    return (output)

def old_get_url_source(url):
    """Returns HTML source at web URL (via lynx)."""
    # TODO: get directly via some python module
    ## command_line = "lynx -source '%s'" % url
    ## command_line = "lynx --useragent=Nonesuch -source '%s'" % url
    # via http://meta.wikimedia.org/wiki/User-Agent_policy
    #     If you run a bot, please send a User-Agent header identifying the bot and supplying some way of contacting you, e.g.:
    #     User-Agent: MyCoolTool/1.1 (http://example.com/MyCoolTool/; MyCoolTool@example.com) BasedOnSuperLib/1.4
    ## command_line = "lynx --useragent='TPO/1.0 (tomasohara@gmail.com)' -source '%s'" % url
    user_agent = "TPO/1.0 (tomasohara@gmail.com): lynx version 2.8.5rel.5 w/ libwww-FM 2.14, SSL-MM 1.4.1, and OpenSSL 0.9.8t"
    command_line = "lynx --useragent='%s' -source '%s'" % (user_agent, url)
    return (run_command(command_line))

def get_url_source(url):
    """Returns HTML source at web URL."""
    debug_print("get_url_source(%s)" % url, 5)
    my_url_opener = MyURLOpener()
    page = my_url_opener.open(url)
    contents = page.read()
    if URL_SLEEP:
        debug_print("Pausing for %d second(s)" % URL_SLEEP, 5)
        time.sleep(URL_SLEEP)
    debug_print("contents: %s" % contents, 6)
    return (contents)

def alt_get_url_source(url, retries=1):
    """Returns HTML source at URL with retries after pausing"""
    tries = retries + 1
    while (tries > 0):
        contents = get_url_source(url)
        if not re.search("Error: ERR_ACCESS_DENIED", contents):
            break
        debug_print("Warning: access denied error; trying again after pause")
        tries -= 1
        time.sleep(5)
    return (contents)

def write_file(filename, text):
    """Outpus TEXT to FILENAME"""
    debug_print("write_file(%s, _)" % filename, 5)
    try:
        with open(filename, 'w') as f:
            f.write(text)
    except:
        print_stderr("Unable to write file '%s': %s" % (filename, sys.exc_info()))
    return

def download_category_articles(url, depth=0):
    """Downloads all wikipedia articles subsumed by category at URL"""
    if url in processed:
        debug_print("URL %s already processed" % URL, 5)
        return
    processed[url] = True

    # Get HTML source and output to file for category
    # TODO: make file name normalization more robust (check all wikipedia category names for problematic characters that commonly occur)
    ## category_source = get_url_source(url)
    category_source = get_url_source(url)
    ## TEST: category_source = alt_get_url_source(url)
    category_name = url
    category_name = re.sub(r'^.*/', "", category_name)
    category_name = re.sub(r'[ :&;]', "_", category_name)
    write_file(category_name + ".html", category_source)

    # Scan through the category definition
    in_subcats_section = False
    in_pages_section = False
    line_num = 0
    for line in category_source.split("\n"):
        debug_print("%sl%d: %s" % (("\t"*depth), line_num, line), 9)
        line_num += 1
        # Update state indicator
        if re.search(r"Subcategories", line):
            in_subcats_section = True
            debug_print("Start of subcategories section at line %d" % line_num, 4)
        elif re.search(PAGES_START, line):
            ## OLD: elif re.search(r"Pages in category", line):
            # TODO: use '<div id="mw-pages">' instead for flagging pages ???
            in_subcats_section = False
            in_pages_section = True
            debug_print("Start of pages section at line %d" % line_num, 4)
        elif re.search(PAGES_END, line):
            ## OLD: elif re.search(r"Media in category", line):
            debug_print("End of pages section at line %d" % line_num, 4)
            in_pages_section = False
        elif my_re.search(r'</table>.*href="([^"]+)".*>.*next (\d+).*</a>', line):
            # note: above use of </table> avoids matching next link prior to pages group
            in_pages_section = False
            continuation_url = base_url + my_re.match.group(1)
            next_num = my_re.match.group(2)
            debug_print("Recursing over URL with next %s entries: %s" % (next_num, continuation_url), 4)
            if not SKIP_SUBCATS:
                download_category_articles(continuation_url, depth=(1+depth))

        # Extract subcategory hyperlink
        if in_subcats_section:
            match = re.search(r'href="(/wiki/Category:[^"]+)"', line)
            if match:
                subcat_name = match.group(1)
                subcat_url = base_url + subcat_name
                debug_print("Recursing over subcat URL: %s" % subcat_url, 4)
                if not SKIP_SUBCATS:
                    if MAKE_SUBDIRS:
                        # Optionally creates new directory for storing pages
                        make_directory(subcat_name)
                        change_directory(subcat_name)
                    download_category_articles(subcat_url, depth=(1+depth))
            else:
                debug_print("Ignoring category-section line: %s" % line, 6)
        # Extract page hyperlinks
        elif in_pages_section:
            match = re.search(r'<li>.*href="(/wiki/([^"]+))"', line)
            if match:
                page_url = base_url + match.group(1)
                article_name = match.group(2)
                debug_print("Downloading article URL: %s" % page_url, 4)
                if not SKIP_PAGES:
                    page_source = get_url_source(page_url)
                    article_name = re.sub(r'[ :]', "_", article_name)
                    write_file(article_name + ".html", page_source)
            else:
                debug_print("Ignoring pages-section line: %s" % line, 6)
        else:
            debug_print("Ignoring misc. line: %s" % line, 6)

def main():
    """
    Main routine: parse arguments and perform main processing
    TODO: revise comments
    Note: Used to avoid conflicts with globals (e.g., if this were done at end of script).
    """
    # 
    debug_print("start %s: %s" % (__file__, debug_timestamp()), 3)

    # Parse command-line, showing usage statement if no arguments given (or --help)
    args = sys.argv
    debug_print("argv = %s" % sys.argv, 3)
    num_args = len(args)
    if ((num_args == 1) or ((num_args > 1) and (args[1] == "--help"))):
        print_stderr("Usage: %s [--help] URL" % args[0])
        print_stderr("Example: %s 'http://en.wikipedia.org/wiki/Category:Major_League_Baseball_players'" % args[0])
        print_stderr("Notes:")
        print_stderr("- MAKE_SUBDIRS recreates category hierarchy in file system")
        print_stderr("- SKIP_SUBCATS disables subcat traversals")
        print_stderr("- SKIP_PAGES disables page downloading")
        sys.exit()
    arg_pos = 1
    while (arg_pos < num_args) and (args[arg_pos][0] == "-"):
        debug_print("args[%d]: %s" % (arg_pos, args[arg_pos]), 3)
        if (args[arg_pos] == "-"):
            # note: - is used to avoid usage statement with file input from stdin
            pass
        else:
            print_stderr("Error: unexpected argument '%s'" % args[arg_pos])
            sys.exit()
        arg_pos += 1
    url = args[arg_pos]

    # Optionally create and change into output directory for wiki pages
    if OUTPUT_DIR != ".":
        make_directory(OUTPUT_DIR)
        change_directory(OUTPUT_DIR)
    
    # Process wiki categorization file
    # TODO: accept file input as well (e.g., '2013 Pittsburgh Pirates season - Wikipedia, the free encyclopedia.html')
    download_category_articles(url)
    debug_print("stop %s: %s" % (__file__, debug_timestamp()), 3)

#------------------------------------------------------------------------
# Standalone processing

if __name__ == '__main__':
    main()
