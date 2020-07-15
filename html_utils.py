#! /usr/bin/env python
#
# Micellaneous HTML utility functions, in particular with support for resolving
# HTML rendered via JavaScript. This was motivated by the desire to extract
# images from pubchem.ncbi.nlm.nih.gov web pages for drugs (e.g., Ibuprofen,
# as illustrated below).
#
#-------------------------------------------------------------------------------
# Example usage:
#
# TODO: see what html_file should be set to
# $ PATH="$PATH:/usr/local/programs/selenium" DEBUG_LEVEL=6 MOZ_HEADLESS=1 $PYTHON html_utils.py "$html_file" > _html-utils-pubchem-ibuprofen.log7 2>&
# $ cd $TMPDIR
# $ wc *ibuprofen*
#     13   65337  954268 post-https%3A%2F%2Fpubchem.ncbi.nlm.nih.gov%2Fcompound%2FIbuprofen
#   3973   24689  178909 post-https%3A%2F%2Fpubchem.ncbi.nlm.nih.gov%2Fcompound%2FIbuprofen.txt
#     60    1152   48221 pre-https%3A%2F%2Fpubchem.ncbi.nlm.nih.gov%2Fcompound%2FIbuprofen
#
# $ count_it.perl "<img" pre-https%3A%2F%2Fpubchem.ncbi.nlm.nih.gov%2Fcompound%2FIbuprofen
# $ count_it.perl "<img" post-https%3A%2F%2Fpubchem.ncbi.nlm.nih.gov%2Fcompound%2FIbuprofen
# <img	7
#
# $ diff pre-https%3A%2F%2Fpubchem.ncbi.nlm.nih.gov%2Fcompound%2FIbuprofen post-https%3A%2F%2Fpubchem.ncbi.nlm.nih.gov%2Fcompound%2FIbuprofen
# ...
# <   <body>
# <     <div id="js-rendered-content"></div>
# ---
# > 
# >     <div id="js-rendered-content"><div class="relative flex-container-vertical"><header class="bckg-white b-bottom"><div class="bckg
# 54c7
# <     <script type="text/javascript" async src="/pcfe/summary/summary-v3.min.js"></script><!--<![endif]-->
# ---
# >     <script type="text/javascript" async="" src="/pcfe/summary/summary-v3.min.js"></script><!--<![endif]-->
# 58,60c11,13
# <     <script type="text/javascript" async src="https://www.ncbi.nlm.nih.gov/core/pinger/pinger.js"></script>
# <   </body>
#
#-------------------------------------------------------------------------------
# TODO:
# - Standardize naming convention for URL parameter accessors (e.g., get_url_param vs. get_url_parameter).
# 

"""HTML utility functions"""

import sys
import time

# Note: selenium import now optional
## OLD: from selenium import webdriver

import debug
import system

# Constants
MAX_DOWNLOAD_TIME = system.getenv_integer("MAX_DOWNLOAD_TIME", 60)
MID_DOWNLOAD_SLEEP_SECONDS = system.getenv_integer("MID_DOWNLOAD_SLEEP_SECONDS", 60)
POST_DOWNLOAD_SLEEP_SECONDS = system.getenv_integer("POST_DOWNLOAD_SLEEP_SECONDS", 0)
SKIP_BROWSER_CACHE = system.getenv_boolean("SKIP_BROWSER_CACHE", False)
USE_BROWSER_CACHE = (not SKIP_BROWSER_CACHE)

# Globals
# note: for convenience in Mako template code
user_parameters = {}

#-------------------------------------------------------------------------------
# Helper functions (TODO, put in system.py)

TEMP_DIR = system.getenv_text("TMPDIR", "/tmp")
#
def write_temp_file(filename, text):
    """Create FILENAME in temp. directory using TEXT"""
    temp_path = system.form_path(TEMP_DIR, filename)
    system.write_file(temp_path, text)
    return

#-------------------------------------------------------------------------------
# HTML utility functions

browser_cache = {}
##
def get_browser(url):
    """Get existing browser for URL or create new one"""
    browser = None
    global browser_cache
    # Check for cached version of browser. If none, create one and access page.
    browser = browser_cache.get(url) if USE_BROWSER_CACHE else None
    if not browser:
        # HACK: unclean import (i.e., buried in function)
        from selenium import webdriver       # pylint: disable=import-error, import-outside-toplevel
        browser = webdriver.Firefox()
        if USE_BROWSER_CACHE:
            browser_cache[url] = browser
        browser.get(url)
        if POST_DOWNLOAD_SLEEP_SECONDS:
            time.sleep(POST_DOWNLOAD_SLEEP_SECONDS)
    # Make sure the bare minimum is included (i.e., "<body></body>"
    debug.assertion(len(browser.execute_script("return document.body.outerHTML")) > 13)
    debug.trace_fmt(5, "get_browser({u}) => {b}", u=url, b=browser)
    return browser


def get_inner_html(url):
    """Return the fully-rendered version of the URL HTML source (e.g., after JavaScript DOM manipulation"""
    # Based on https://stanford.edu/~mgorkove/cgi-bin/rpython_tutorials/Scraping_a_Webpage_Rendered_by_Javascript_Using_Python.php
    # Navigate to the page (or get browser instance with existing page)
    browser = get_browser(url)
    # Wait for Javascript to finish processing
    wait_until_ready(url)
    # Extract fully-rendered HTML
    inner_html = browser.execute_script("return document.body.innerHTML")
    debug.trace_fmt(7, "get_inner_html({u}) => {h}", u=url, h=inner_html)
    return inner_html


def get_inner_text(url):
    """Get text of URL after JavaScript processing"""
    # See https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement/innerText
    # Navigate to the page (or get browser instance with existing page)
    browser = get_browser(url)
    # Wait for Javascript to finish processing
    wait_until_ready(url)
    # Extract fully-rendered text
    inner_text = browser.execute_script("return document.body.innerText")
    debug.trace_fmt(7, "get_inner_text({u}) => {t}", u=url, h=inner_text)
    return inner_text


def document_ready(url):
    """Determine whether document for URL has completed loading"""
    # See https://developer.mozilla.org/en-US/docs/Web/API/Document/readyState
    browser = get_browser(url)
    ready_state = browser.execute_script("return document.readyState")
    is_ready = (ready_state == "complete")
    debug.trace_fmt(6, "document_ready({u}) => {r}; state={s}",
                    u=url, r=is_ready, s=ready_state)
    return is_ready


def wait_until_ready(url):
    """Wait for document ready and pause to allow loading to finish"""
    # TODO: make sure the sleep is proper way to pause
    debug.trace_fmt(5, "in wait_until_ready({u})", u=url)
    start_time = time.time()
    end_time = start_time + MAX_DOWNLOAD_TIME
    while ((start_time < end_time) and (not document_ready(url))):
        time.sleep(MID_DOWNLOAD_SLEEP_SECONDS)
        if (not document_ready(url)):
            debug.trace_fmt(5, "Warning: time out ({s} secs) in accessing URL '{u}')'", s=system.round_num(end_time - start_time, 1), u=url)        
    debug.trace_fmt(5, "out wait_until_ready(); elapsed={t}s",
                    t=(time.time() - start_time))
    return
    

def escape_html_value(value):
    """Escape VALUE for HTML embedding"""
    return system.escape_html_text(value)

def unescape_html_value(value):
    """Undo escaped VALUE for HTML embedding"""
    return system.unescape_html_text(value)


def escape_hash_value(hash_table, key):
    """Wrapper around escape_html_value for HASH_TABLE[KEY] (or "" if missing).
    Note: newlines are converted into <br>'s."""
    escaped_item_value = escape_html_value(hash_table.get(key, ""))
    escaped_value = escaped_item_value.replace("\n", "<br>")
    debug.trace_fmtd(7, "escape_hash_value({h}, {k}) => '{r}'", h=hash_table, k=key, r=escaped_value)
    return escaped_value


def get_param_dict(param_dict=None):
    """Returns parameter dict using PARAM_DICT if non-Null else USER_PARAMETERS
       Note: """
    return (param_dict if param_dict else user_parameters)

def set_param_dict(param_dict):
    """Sets global user_parameters to value of PARAM_DICT"""
    global user_parameters
    user_parameters = param_dict

def get_url_param(name, default_value="", param_dict=None):
    """Get value for NAME from PARAM_DICT (e.g., USER_PARAMETERS), using DEFAULT_VALUE (normally "").
    Note: It will be escaped for use in HTML."""
    param_dict = get_param_dict(param_dict)
    value = escape_html_value(param_dict.get(name, default_value))
    value = system.to_unicode(value)
    debug.trace_fmtd(4, "get_url_param({n}, [{d}]) => {v})",
                     n=name, d=default_value, v=value)
    return value
#
get_url_parameter = get_url_param

def get_url_param_checkbox_spec(name, default_value="", param_dict=None):
    """Get value of boolean parameters formatted for checkbox (i.e., 'checked' iff True or on) from PARAM_DICT"""
    # TODO: implement in terms of get_url_param
    param_dict = get_param_dict(param_dict)
    param_value = param_dict.get(name, default_value)
    param_value = system.to_unicode(param_value)
    value = "checked" if (param_value in [True, "on"]) else ""
    debug.trace_fmtd(4, "get_url_param_checkbox_spec({n}, [{d}]) => {v})",
                     n=name, d=default_value, v=value)
    return value
#
get_url_parameter_checkbox_spec = get_url_param_checkbox_spec

def get_url_parameter_bool(param, default_value=False, param_dict=None):
    """Get boolean value for PARAM from PARAM_DICT, with "on" treated as True. @note the hash defaults to user_parameters, and the default value is False"""
    # TODO: implement in terms of get_url_param
    param_dict = get_param_dict(param_dict)
    ## OLD:
    result = (param_dict.get(param, default_value) in ["on", True])
    ## HACK: result = ((system.to_unicode(param_dict.get(param, default_value))) in ["on", True])
    debug.trace_fmtd(4, "get_url_parameter_bool({p}, {dft}, _) => {r}",
                     p=param, dft=default_value, r=result)
    return result
#
get_url_param_bool = get_url_parameter_bool

#-------------------------------------------------------------------------------

def main(args):
    """Supporting code for command-line processing"""
    debug.trace_fmtd(6, "main({a})", a=args)
    user = system.getenv_text("USER")
    system.print_stderr("Warning, {u}: Not really intended for direct invocation".format(u=user))

    # HACK: Do simple test of inner-HTML support
    if (len(args) > 1):
        debug.trace_fmt(4, "browser_cache: {bc}", bc=browser_cache)
        url = args[1]
        html_data = system.download_web_document(url)
        filename = system.quote_url_text(url)
        if debug.debugging():
            write_temp_file("pre-" + filename, html_data)
        ## OLD: wait_until_ready(url)
        ## BAD: rendered_html = render(html_data)
        rendered_html = get_inner_html(url)
        if debug.debugging():
            write_temp_file("post-" + filename, rendered_html)
        print("Rendered html:")
        print(system.to_utf8(rendered_html))
        if debug.debugging():
            rendered_text = get_inner_text(url)
            debug.trace_fmt(5, "type(rendered_text): {t}", t=rendered_text)
            write_temp_file("post-" + filename + ".txt", rendered_text)
        debug.trace_fmt(4, "browser_cache: {bc}", bc=browser_cache)
    else:
        print("Specify a URL as argument 1 for a simple test of inner access")
    return

if __name__ == '__main__':
    main(sys.argv)
