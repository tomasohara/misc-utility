#! /usr/bin/env python
#
# Miscellaneous text utility functions, such as for extracting text from
# HTML and other documents.
#
# Notes:
# - Modules in other directoriess rely upon this (e.g., ~/visual-search),
#   so do global search before making significant changes.
# - This is in the spirit of quick-n-dirty (e.g., for R&D: they are packages
#   that are better suited for industrial strength code (e.g., for production).
#
# TODO:
# - Write up test suite, el tonto!.
# - Add pointer to specific packages better for production use.
# - Move HTML-specific functions into html_utils.py.
#

"""Miscellaneous text utility functions"""

# Library packages
import re
import six
import sys

# Installed packages
## hACK: temporarily make these conditional
## from bs4 import BeautifulSoup
## import textract

# Local packages
import debug
from regex import my_re
import system
from system import to_int

# TEMP: Placeholders for dynamically loaded modules
BeautifulSoup = None
textract = None

def init_BeautifulSoup():
    """Make sure bs4.BeautifulSoup is loaded"""
    import bs4                           # pylint: disable=import-outside-toplevel, import-error
    global BeautifulSoup
    BeautifulSoup = bs4.BeautifulSoup


def html_to_text(document_data):
    """Returns text version of html DATA"""
    # EX: html_to_text("<html><body><!-- a cautionary tale -->\nMy <b>fat</b> dog has fleas</body></html>") => "My fat dog has fleas"
    # Note: stripping javascript and style sections based on following:
    #   https://stackoverflow.com/questions/22799990/beatifulsoup4-get-text-still-has-javascript
    debug.trace_fmtd(7, "html_to_text(_):\n\tdata={d}", d=document_data)
    ## OLD: soup = BeautifulSoup(document_data)
    init_BeautifulSoup()
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


def init_textract():
    """Make sure textract is loaded"""
    import textract as ex                # pylint: disable=import-outside-toplevel, import-error
    global textract
    textract = ex


def document_to_text(doc_filename):
    """Returns text version of document FILENAME of unspecified type"""
    text = ""
    try:
        init_textract()
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
    init_BeautifulSoup()
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
    for image in all_images:
        debug.trace_fmtd(6, "image={inf}; style={sty}", inf=image, sty=image.attrs.get('style'))
        ## TEST: if (image.has_attr('attrs') and (image.attrs.get['style'] in ["display:none", "visibility:hidden"])):
        if (image.attrs.get('style') in ["display:none", "visibility:hidden"]):
            debug.trace_fmt(5, "Ignoring hidden image: {img}", img=image)
            continue
        image_src = image.get("src", "")
        if not image_src:
            debug.trace_fmt(5, "Ignoring image without src: {img}", img=image)
            continue
        if image_src.startswith("/"):
            image_src = web_site_url + image_src
        elif not image_src.startswith("http"):
            image_src = base_url + "/" + image_src
        images.append(image_src)
    debug.trace_fmtd(6, "extract_html_images() => {i}", i=images)
    return images


def version_to_number(version, max_padding=3):
    """Converts VERSION to number that can be used in comparisons
    Note: The Result will be of the form M.mmmrrrooo..., where M is the
    major number m is the minor, r is the revision and o is other.
    Each version component will be prepended with up MAX_PADDING [3] 0's
    Notes:
    - strings in the version are ignored
    - 0 is returned if version string is non-standard"""
    # EX: version_to_number("1.11.1") => 1.00010001
    # EX: version_to_number("1") => 1
    # EX: version_to_number("") => 0
    # TODO: support string (e.g., 1.11.2a).
    version_number = 0
    version_text = version
    new_version_text = ""
    max_component_length = (1 + max_padding)
    debug.trace_fmt(5, "version_to_number({v})", v=version)

    # Remove all alphabetic components
    version_text = re.sub(r"[a-z]", "", version_text, re.IGNORECASE)
    if (version_text != version):
        debug.trace_fmt(2, "Warning: stripped alphabetic components from version: {v} => {nv}", v=version, nv=version_text)

    # Remove all spaces (TODO: handle tabs and other whitespace)
    version_text = version_text.replace(" ", "")

    # Convert component numbers iteratively and zero-pad if necessary
    # NOTE: Components greater than max-padding + 1 treated as all 9's.
    debug.trace_fmt(4, "version_text: {vt}", vt=version_text)
    first = False
    num_components = 0
    regex = r"^(\d+)(\.((\d*).*))?$"
    while (my_re.search(regex, version_text)):
        component = my_re.group(1)
        # TODO: fix my_re.group to handle None as ""
        version_text = my_re.group(2) if my_re.group(2) else ""
        num_components += 1
        debug.trace_fmt(4, "new version_text: {vt}", vt=version_text)

        component = system.to_string(system.to_int(component))
        if first:
            new_version_text = component + "."
            regex = r"^(\d+)\.?((\d*).*)$"
        else:
            if (len(component) > max_component_length):
                old_component = component
                component = "9" * max_component_length
                debug.trace_fmt(2, "Warning: replaced overly long component #{n} {oc} with {c}",
                                n=num_components, oc=old_component, nc=component)
            new_version_text += component
            debug.trace_fmt(4, "Component {n}: {c}", n=num_components, c=component)
    version_number = system.to_float(new_version_text, version_number)
    ## TODO:
    ## if (my_re.search(p"[a-z]", version_text, re.IGNORECASE)) {
    ##     version_text = my_re.... 
    ## }
    debug.trace_fmt(4, "version_to_number({v}) => {n}", v=version, n=version_number)
    return version_number


def extract_string_list(text):
    """Extract list of string values in TEXT string separated by spacing or a comma.
    Note: the string values currently cannot be quoted (i.e., no embedded spaces)."""
    # EX: extract_string_list("1, 2,3 4") => ['1', '2', '3', '4']
    # EX: extract_string_list(" ") => []
    # TODO: add support for quoted items (e.g., "'my dog', 'likes', 'no cats' ")
    normalized_text = text.replace(",", " ").strip()
    value_list = re.split(" +", normalized_text)
    if (value_list == [""]):
        value_list = []
    debug.assertion("" not in value_list)
    debug.trace_fmtd(5, "extract_string_list({t}) => {vl}", t=text, vl=value_list)
    return value_list


def extract_int_list(text, default_value=0):
    """Extract list of integral values from comma and/or whitespace delimited TEXT using DEFAULT_VALUE for non-integers (even if floating point)"""
    return [to_int(v, default_value) for v in extract_string_list(text)]


def getenv_ints(var, default_values_spec):
    """Get integer list using values specified for environment VAR (or DEFAULT_VALUES_SPEC)"""
    # EX: getenv_ints("DUMMY VARIABLE", str(list(range(5)))) => [0, 1, 2, 3, 4]
    return extract_int_list(system.getenv_text(var, default_values_spec))


def is_symbolic(token):
    """Indicates whether (string) token is symbolic (e.g., non-numeric).
    Note: for convenience, tokens can be numeric types, with False returned."""
    # EX: is_symbolic("PI") => True
    # EX: is_symbolic("3.14159") => False
    # EX: is_symbolic("123") => False
    # EX: is_symbolic(123) => False
    # EX: is_symbolic(0.1) => False
    # TODO: add support for complex numbers
    in_token = token
    result = True
    try:
        if isinstance(token, six.string_types):
            token = token.strip()
        # Note: if an exception is not triggered, the token is numeric
        if (float(token) or int(token)):
            debug.trace(6, "is_symbolic: '{t}' is numeric".format(t=token))
        result = False
    except (TypeError, ValueError):
        pass
    debug.trace_fmt(7, "is_symbolic({t}) => {r})", t=in_token, r=result)
    return result


#-------------------------------------------------------------------------------

if __name__ == '__main__':
    system.print_stderr("Error: not intended for command-line use")
    debug.assertion("html" not in html_to_text("<html><body></body></html>"))
