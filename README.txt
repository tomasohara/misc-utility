Miscellaneous utility Python scripts developed over the course of several
independent consulting projects (e.g., Mega Forte Intemass and Juju Job Search).
Note that in some cases, newer versions of the script use the tpo_ prefix, which
was due to an expedient to avoid creation of a separate namespace.

The software is licensed under the GNU Lesser General Public Version 3 (LGPLv3). See LICENSE.txt.

Synopsis for each script:
- add_javascript_tracing.py: Add tracing to Javascript functions via console.debug.
- check-grammar.py: run text through grammar checker (e.g., the one from OpenOffice).
- check_time_tracking.py: Sanity check for time-tracking report.
- chunk_text.py: Runs text through chunker, grouping sets of words based on related tags.
- common.py: Python module with common utilities mostly for debugging.
- compute_tfidf.py: compute Term Frequency Inverse Document Frequency (TF-IDF).
- debug.py: Functions for debugging, such as console tracing.
- download_wiki_category.py: download-wiki-category.py: downloads all Wikipedia articles in category (and descendant cateories).
- encoding_converter.py: unicode-converter.py: converts between various file formats (e.g., Windows-1259 to UTF-8).
- extract_document_text.py: extract_document_text.py: extract text from documents of various types.
- extract_wiki_texts.py: extract_wiki_token.py: Extracts tokens from each of the articles in a Wikipedia dump.
- format_profile.py: displays the result of python profiling via cProfile.
- glue_helpers.py: Utility functions for writing glue scripts, such as implementing functionality.
- merge_notes.py: merge_notes.py: Merge textual note files based on timestamps.
- merge_tabular_data.py: Merge the contents of two tabular files with tab-separated values.
- misc_utils.py: Miscellaneous functions not suitable for other modules (e.g., system.py).
- ngram_tfidf.py: Support for performing Term Frequency (TF) Inverse Document Frequency (IDF).
- parse-text.py: Run text through grammatical parser (e.g., Stanford).
- randomize-lines.py: Randomize lines in a file without reading entirely into memory.
- regex.py: Convenience class for regex searching, providing simple wrapper around static match results.
- sys_version_info_hack.py: Hack for redefining sys.version_info to be python3 compatibility.
- system.py: Functions for system related access, such as running command or
- text_categorizer.py: Class for text categorizer using Scikit-Learn.
- text_processing.py: Performs text processing (e.g., via NLTK).
- text_utils.py: Miscellaneous text utility functions, such as for extracting text from such as for extracting text from HTML and other documents.
- tpo_common.py: Python module with common utilities mostly for debugging (by Tom O'Hara)
- tpo_text_processing.py: Performs text processing (e.g., via NLTK).
- train_text_categorizer.py: Trains text categorizer using Scikit-Learn.
- wordnet.py: Module for WordNet access using NLTK WordNet package.

TODO:
- Update script synopsis's based on the above.

Thomas P. O'Hara
Decmeber 2018
