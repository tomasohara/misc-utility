#! /usr/bin/env python
#
# Customization of PyLucene IndexFiles.py to index each line in table file as a
# mini-document:
#    http://svn.apache.org/viewvc/lucene/pylucene/trunk/samples/IndexFiles.py
# This also includes an option to do similar indexing via pysolr.
#
# Notes:
# - By default, this uses the fields 'name' for document id and 'contents' for data,
#   where the line is assumed to be in the format name<TAB>content.
# - Alternatively, the line number is used for the 'name' field and entire line for 'contents'.
# - Term vectors can optionally be added to the index to support TF/IDF analysis.
# - The Solr implementation is not as comprehensive as the Lucene support (e.g., term vectors not yet supported).
#
# TODO:
# - Replace 'name' with 'id'.
# - Add option for whitespace tokenizer (e.g., to better match R&D prototype).
# - Add progress indicator.
# - ** Add simple example(s) to usage **
# - Reimplement Lucene-vs-Solr support via subclassing, which would be
#   cleaner but a bit tedious (e.g., given monolithic methods like indexTable).
#

"""
Usage: python {script} table_file [index_dir] [append]

This script is loosely based on the Lucene demo sample IndexFiles.java
(i.e., org.apache.lucene.demo.IndexFiles). Instead of traversing a directory
structure for the documents, a single file is used with the contents on
separate lines. The resulting Lucene index is placed in the current directory 
and called '{index_dir}'.

Notes:
- The index directory is relative to current directory (not script directory 
  as in original script).
- The processing can be customized via the following environment variables:
  ID_FIELD: field number for mini-document name/id (1-based)
  INCLUSION_FIELDS: comma-separated field numbers for mini-document content
  SHOW_TFIDF: to include term vectors for TF/IDF support (boolean or 0/1)
  IMPLICIT_ID: use the line number as the ID, not first field (boolean or 0/1)
  USE_SOLR: whether to use a Solr server instead for the indexing
  SOLR_URL: the URL to use to connect to the Solr server

Example:
$ {script} tests/random100-titles-descriptions.txt
$ ls table-file-index
"""
# TODO: STORE_CONTENTS, SHOW_TFIDF, USE_NGRAMS, APPEND_INDEX

import logging
import os
import sys
import threading
import time
from datetime import datetime
import logging

import tpo_common as tpo
from tpo_common import getenv_text, getenv_integer, getenv_number, getenv_boolean, debug_print, debug_format
import glue_helpers as gh

USE_SOLR = getenv_boolean("USE_SOLR", False)
if USE_SOLR:
    import pysolr
else:
    import lucene
    from java.io import File
    from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
    from org.apache.lucene.analysis.standard import StandardAnalyzer
    from org.apache.lucene.document import Document, Field, FieldType
    from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig
    from org.apache.lucene.store import SimpleFSDirectory
    from org.apache.lucene.util import Version

#------------------------------------------------------------------------
# Globals

DEFAULT_INDEX_DIR = "table-file-index"
INDEX_DIR = getenv_text("INDEX_DIR", DEFAULT_INDEX_DIR, "Location for Lucene index")
# Determine fields to use for mini-document name (i.e., ID) and for optional doc. title
ID_FIELD = (tpo.getenv_integer("ID_FIELD", 1) - 1)	# note: same as NAME
TITLE_FIELD = (tpo.getenv_integer("TITLE_FIELD", 0) - 1)
# Determine fields from table to include (1-based spec. in INCLUSION_FIELDS)
INCLUSION_FIELDS = getenv_text("INCLUSION_FIELDS", "")
ALL_FIELDS = (INCLUSION_FIELDS == "")
INCLUDE_FIELD = {}
SHOW_TFIDF = getenv_boolean("SHOW_TFIDF", False)
IMPLICIT_ID = getenv_boolean("IMPLICIT_ID", False)
assert(IMPLICIT_ID or (ID_FIELD >= 0))
ENSURE_UNICODE = getenv_boolean("ENSURE_UNICODE", False)
APPEND_INDEX = getenv_boolean("APPEND_INDEX", False)
STORE_CONTENTS = getenv_boolean("STORE_CONTENTS", False)
USE_NGRAMS = getenv_boolean("USE_NGRAMS", False, "Use ngram tokens")
MIN_NGRAM = getenv_integer("MIN_NGRAM", 2, "Minimum umber of ngram tokens (besides unigrams)")
MAX_NGRAM = getenv_integer("MAX_NGRAM", 3, "Maximum number of ngrams tokens")
DEFAULT_SOLR_URL = "http://localhost:8983/solr"
SOLR_URL = getenv_text("SOLR_URL", DEFAULT_SOLR_URL)
SOLR_BATCH_SIZE = tpo.getenv_integer("SOLR_BATCH_SIZE", 128)

# Initialize the inclusion field list
# TODO: put in script initialization section (e.g., at end)
def include_field(num):
    """Enables indexing of input field NUM (1-based)"""
    INCLUDE_FIELD[num - 1] = True
if not ALL_FIELDS:
    map(lambda num: include_field(int(num)), INCLUSION_FIELDS.split(","))
    debug_print("INCLUDE_FIELD: %s" % INCLUDE_FIELD, 4)

#------------------------------------------------------------------------
# Miscellaneous initialization

if USE_NGRAMS:
    from org.apache.lucene.analysis.shingle import ShingleAnalyzerWrapper

#------------------------------------------------------------------------

class Ticker(object):
    """Class for in-progress ticker. This needs to be run as a thread."""

    def __init__(self):
        """Constructor"""
        debug_print("Ticker.__init__()", 5)
        self.tick = True

    def run(self):
        """Main thread routine"""
        debug_print("Ticker.run()", 5)
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)

class IndexTableFile(object):
    """Class for indexing rows in a tabular file as mini-documents"""

    def __init__(self, tableFile, storeDir, analyzer=None):
        """Constructor that indexes TABLEFILE, placing index under STOREDIR and using Lucene ANALYZER"""
        debug_format("IndexTableFile.__init__({tableFile}, {storeDir}, {analyzer})", 5)

        if USE_SOLR:
            # TODO: support APPEND_INDEX option and add option for timeout
            # format: Solr(url, [decoder=None], [timeout=60])
            self.solr = pysolr.Solr(SOLR_URL)
            tpo.debug_print("solr=%s" % self.solr, 4)
            writer = None
            # TODO: add abity to create new collections dynamically
            # NOTE: as workaorund alternative collections are reserved (e.g., collection2 ... collection9) via cut-n-paste of collection1
            # Add sanity check for different collection URL (e.g. http://localhost:8983/solr/collection2)
            if (INDEX_DIR != DEFAULT_INDEX_DIR):
                gh.assertion(SOLR_URL != DEFAULT_SOLR_URL)
            #
            # Enable detailed Solr tracing if debugging
            if (tpo.verbose_debugging()):
                ## solr_logger = logging.getLogger('pysolr')
                ## solr_logger.setLevel(logging.DEBUG)
                # TODO: use mapping from symbolic LEVEL user option (e.g., via getenv)
                level = logging.INFO if (tpo.debug_level < 4) else logging.DEBUG
                logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=level)

        else:
            # TODO: why is java.awt.headless enabled?
            lucene.initVM(vmargs=['-Djava.awt.headless=true'])
            print('lucene: %s' % lucene.VERSION)

            # Create index directory if needed
            if not os.path.exists(storeDir):
                debug_format("Creating index directory: {storeDir}", 4)
                os.mkdir(storeDir)

            # Instantiate a standard analyzer if needed
            if not analyzer:
                analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
                tpo.debug_format("analyzer={analyzer}", 5)
            store = SimpleFSDirectory(File(storeDir))
            # note: allows up to 1M tokens per field
            analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
            if USE_NGRAMS:
                shingle_wrapper = ShingleAnalyzerWrapper(analyzer, MIN_NGRAM, MAX_NGRAM)
                debug_print("shingle_wrapper=%s" % shingle_wrapper, 5)
                analyzer = shingle_wrapper
            config = IndexWriterConfig(Version.LUCENE_CURRENT, analyzer)
            open_mode = IndexWriterConfig.OpenMode.CREATE_OR_APPEND if APPEND_INDEX else IndexWriterConfig.OpenMode.CREATE
            config.setOpenMode(open_mode)
            writer = IndexWriter(store, config)

        self.indexTable(tableFile, writer)
        ticker = Ticker()
        sys.stdout.write('commiting index: ')
        threading.Thread(target=ticker.run).start()
        if USE_SOLR:
            # TODO?: self.solr.optimize()
            pass
        else:
            writer.commit()
            writer.close()
        ticker.tick = False
        print('done')

    def indexTable(self, tableFile, writer):
        """Indexes TABLEFILE using Lucene index WRITER (ignored if USE_SOLR)"""
        # Note: The content field normally is not stored, but this is useful for debugging
        # with Luke (i.e., Lucene Index Toolkit). This is also useful for TF/IDF diagnostics
        # (see get_doc_terms_and_freqs in search_table_file_index.py).
        debug_format("indexTable({tableFile}, {writer})", 5)

        # Specify indexing field types
        if USE_SOLR:
            # TODO: enable corresponding Solr options (e.g., storing contents field)
            all_records = []
        else:
            name_field_type = FieldType()
            name_field_type.setIndexed(True)
            name_field_type.setStored(True)
            name_field_type.setTokenized(True)
            name_field_type.setIndexOptions(FieldInfo.IndexOptions.DOCS_AND_FREQS)
            #
            if (TITLE_FIELD >= 0):
                title_field_type = FieldType()
                title_field_type.setIndexed(True)
                title_field_type.setStored(True)
                title_field_type.setTokenized(True)
                title_field_type.setIndexOptions(FieldInfo.IndexOptions.DOCS_AND_FREQS)
            #
            content_field_type = FieldType()
            content_field_type.setIndexed(True)
            content_field_type.setStored(STORE_CONTENTS)
            content_field_type.setTokenized(True)
            content_field_type.setIndexOptions(FieldInfo.IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        # Add optional support to support TF/IDF (term vectors, etc.)
        if SHOW_TFIDF:
            if USE_SOLR:
                tpo.print_stderr("Warning: TF/IDF not supported")
            else:
                content_field_type.setStoreTermVectors(True)
                # TDOO: use bare minimum term vector settings
                content_field_type.setStoreTermVectorOffsets(True)
                content_field_type.setStoreTermVectorPositions(True)
                debug_print("Enabled TF/IDF support (e.g., term vectors)", 4)

        # Extract specified fields from tabular input, creating mini-document for each row
        with open(tableFile) as f:
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip("\n")
                debug_print("L%d: %s" % (line_num, line), 6)
                if ENSURE_UNICODE:
                    debug_print("pre type(line): %s" % type(line), 7)
                    line = unicode(line, 'utf-8', errors='replace')
                    debug_print("post type(line): %s" % type(line), 7)
                try:
                    # Derive field contexts
                    # Note: doc ID is 0-based (to match Lucene docno).
                    # TODO: Omit NAME field when IMPLICIT_ID used.
                    title = ""
                    name = str(line_num - 1)
                    contents = line
                    # TODO: assume IMPLICIT_ID if no tab in line
                    fields = line.split("\t")
                    if not IMPLICIT_ID:
                        name = fields[ID_FIELD]
                    if (TITLE_FIELD >= 0):
                        title = fields[TITLE_FIELD]
                    if not ALL_FIELDS:
                        contents = ""
                        for f in range(0, len(fields)):
                            if (f in INCLUDE_FIELD):
                                if contents:
                                    contents += "; "
                                contents += fields[f]

                    # Add to index
                    debug_print("Adding mini-document for line %d: name/id='%s' title='%s' contents='%s'\n" % (line_num, name, title, gh.elide(contents)), 5)
                    if USE_SOLR:
                        # Note: Solr typically uses 'id' for key and 'content' for general text field.
                        # See schema.xml in Solr's configuration directory (e.g., under example/solr/collection1/conf).
                        field_dict = {'id': name,
                                      'content': contents}
                        if title:
                            field_dict['title'] = title
                        ## OLD: self.solr.add([field_dict])
                        # Send current batched-up records if needed
                        if (len(all_records) > SOLR_BATCH_SIZE):
                            tpo.debug_print("Sending Solr batch", 5)
                            # format: add(docs, [commit=True], [boost=None], [commitWithin=None], [waitFlush=None], [waitSearcher=None])
                            self.solr.add(all_records)
                            all_records = []
                        # Add to batch
                        all_records.append(field_dict)
                    else:
                        doc = Document()
                        doc.add(Field("name", name, name_field_type))
                        if title:
                            doc.add(Field("title", title, title_field_type))
                        doc.add(Field("contents", contents, content_field_type))
                        writer.addDocument(doc)
                except:
                    tpo.debug_raise()
                    tpo.print_stderr("Exception in indexTable at line %d: %s" % (line_num, str(sys.exc_info())))

        # Send any remaining batches
        if USE_SOLR and (len(all_records) > 0):
            tpo.debug_print("Sending final Solr batch", 5)
            try:
                self.solr.add(all_records)
            except:
                tpo.debug_raise()
                tpo.print_stderr("Exception in indexTable at line %d: %s" % (line_num, str(sys.exc_info())))
        return

#------------------------------------------------------------------------

def main():
    """Entry point for script"""
    # Check command-line arguments
    # TODO: rework via argparse
    if (len(sys.argv) < 2) or ("--help" in sys.argv):
        print(__doc__.format(script=sys.argv[0], index_dir=INDEX_DIR))
        sys.exit(1)
    table_file = sys.argv[1]
    index_dir = INDEX_DIR
    if (len(sys.argv) > 2):
        index_dir = sys.argv[2]
    if (len(sys.argv) > 3):
        APPEND_INDEX = (sys.argv[3].lower() == "false") or (sys.argv[3] == "0")

    # Enable logging if debugging
    if (tpo.debugging_level()):
        # TODO: use mapping from symbolic LEVEL user option (e.g., via getenv)
        level = logging.INFO if (tpo.debug_level < 4) else logging.DEBUG
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=level)

    # Perform the indexing
    start = datetime.now()
    try:
        IndexTableFile(table_file, index_dir)
        end = datetime.now()
        print("elapsed time: %s" % (end - start))
    except Exception:
        tpo.print_stderr("Exception in IndexTableFile: %s" % str(sys.exc_info()))
        raise
    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
