#! /usr/bin/env python
#
# Miscellaneous text utility functions, such as for extracting text from
# HTML and other documents.
#
# Copyright (c) 2012-2018 Thomas P. O'Hara
#

"""Miscellaneous text utility functions"""

# Library packages
import re
import sys

# Installed packages
from bs4 import BeautifulSoup
import textract

# Local packages
import debug
import system

def html_to_text(document_data):
    """Returns text version of html DATA"""
    # EX: html_to_text("<html><body><!-- a cautionary tale -->\nMy <b>fat</b> dog has fleas</body></html>") => "My fat dog has fleas"
    # Note: stripping javascript and style sections based on following:
    #   https://stackoverflow.com/questions/22799990/beatifulsoup4-get-text-still-has-javascript
    debug.trace_fmtd(7, "html_to_text(_):\n\\tdata={d}", d=document_data)
    ## OLD: soup = BeautifulSoup(document_data)
    soup = BeautifulSoup(document_data, "lxml")
    # Remove all script and style elements
    for script in soup(["script", "style"]):
        # *** TODO: soup = soup.extract(script)
        # -or- Note the in-place change (i.e., destructive).
        script.extract()
    # Get the text
    text = soup.get_text()
    debug.trace_fmtd(6, "html_to_text() => {t}", t=text)
    return text

def document_to_text(doc_filename):
    """Returns text version of document FILENAME of unspecified type"""
    text = ""
    try:
        text = system.from_utf8(textract.process(doc_filename))
    except:
        debug.trace_fmtd(3, "Warning: problem converting document file {f}: {e}",
                         f=doc_filename, e=sys.exc_info())
    return text

def extract_html_images(document_data, url):
    """Returns list of all images in HTML DOC from URL (n.b., URL used to determine base URL)"""
    debug.trace_fmtd(8, "extract_html_images(_):\n\tdata={d}", d=document_data)
    # TODO: add example; return dimensions
    # TODO: have URL default to current directory

    # Parse HTML, extract base URL if given and get website from URL.
    soup = BeautifulSoup(document_data, 'html.parser')
    web_site_url = re.sub(r"(https?://[^\/]+)/?.*", r"\1", url)
    debug.trace_fmtd(6, "wsu1={wsu}", wsu=web_site_url)
    if not web_site_url.endswith("/"):
        web_site_url += "/"
        debug.trace_fmtd(6, "wsu2={wsu}", wsu=web_site_url)
    base_url_info = soup.find("base")
    base_url = base_url_info.get("href") if base_url_info else None
    debug.trace_fmtd(6, "bu1={bu}", bu=base_url)
    if not base_url:
        # TODO: comment and example
        base_url = re.sub(r"(^.*/[^\/]+/)[^\/]+$", r"\1", url)
        debug.trace_fmtd(6, "bu2={bu}", bu=base_url)
    if not base_url:
        base_url = web_site_url
        debug.trace_fmtd(6, "bu3={bu}", bu=base_url)
    if not base_url.endswith("/"):
        base_url += "/"
        debug.trace_fmtd(6, "bu4={bu}", bu=base_url)

    # Get images and resolve to full URL (TODO: see if utility for this)
    # TODO: include CSS background images
    images = []
    all_images = soup.find_all('img')
    for image_info in all_images:
        debug.trace_fmtd(6, "image_info = {inf}", inf=image_info)
        image = image_info.get("src", "")
        if not image:
            continue
        if image.startswith("/"):
            image = web_site_url + image
        elif not image.startswith("http"):
            image = base_url + "/" + image
        image = image_info.get("src", "")
        if not image:
            continue
        if image.startswith("/"):
            image = web_site_url + image
        elif not image.startswith("http"):
            image = base_url + "/" + image
        images.append(image)
    debug.trace_fmtd(6, "extract_html_images() => {i}", i=images)
    return images


#-------------------------------------------------------------------------------

if __name__ == '__main__':
    system.print_stderr("Error: not intended for command-line use")
