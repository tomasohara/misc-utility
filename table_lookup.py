#! /usr/bin/env python
#
# Support for table lookup of named entities and other word lists. This
# supports incremental lookup as with tries or "slot hashes" to avoid having to do repeated
# lookups with similar prefixes.
#
# Notes:
# - Slot hashes are based on the approach used in the GATE NLP toolkit. See
#      http://gate.ac.uk/sale/tao/splitch13.html
# - Includes workaround for picking different classes defined in same file:
#      http://stefaanlippens.net/pickleproblem
# - An empty key is added to lookup the label for the data, which defaults
#   to 'Value' if not specified (as with boolean indicator tables).
#
# TODO:
# - Test Bloom filters as alternative to slot hashes.
# - Look into hybrid hash/trie approach with latter only used for high-frequency phrases.
# - Define iterator and other common collection methods.
# - Make the interface more consistent (e.g., make all constructors accept filename for loading).
#

"""Generic table lookup using a variety of formats"""

import argparse
import os
import re
import sys
from abc import ABCMeta, abstractmethod
import tpo_common as tpo
import glue_helpers as gh

# TODO: Add descriptions for important options
TABLE_TYPES = ["slot-hash", "gate-hash", "char-trie", "pat-trie", "trie", "shelve", "kyoto"]
TABLE_TYPE = tpo.getenv_text("TABLE_TYPE", "slot-hash")
assert(TABLE_TYPE in TABLE_TYPES)
USE_SLOT_HASH = tpo.getenv_boolean("USE_SLOT_HASH", (TABLE_TYPE == "slot-hash"))
USE_CHAR_TRIE = tpo.getenv_boolean("USE_CHAR_TRIE", (TABLE_TYPE == "char-trie"))
USE_PATRICIA_TRIE = tpo.getenv_boolean("USE_PATRICIA_TRIE", (TABLE_TYPE == "pat-trie"))
USE_TRIE = tpo.getenv_boolean("USE_TRIE", (TABLE_TYPE.endswith("trie")) or USE_CHAR_TRIE or USE_PATRICIA_TRIE)
MAX_PHRASE_LEN = tpo.getenv_integer("MAX_PHRASE_LEN", 25)
USE_PHRASE_SLOTS = tpo.getenv_boolean("USE_PHRASE_SLOTS", (TABLE_TYPE == "gate-hash"))
USE_WORD_SLOTS = (not USE_PHRASE_SLOTS)
USE_SHELVE = tpo.getenv_boolean("USE_SHELVE", (TABLE_TYPE == "shelve"))
USE_KYOTO = tpo.getenv_boolean("USE_KYOTO", (TABLE_TYPE == "kyoto"))
USE_DB_HASH = (USE_SHELVE or USE_KYOTO)
BRUTE_FORCE = tpo.getenv_boolean("BRUTE_FORCE", USE_DB_HASH)
verbose = tpo.getenv_boolean("VERBOSE", False)

SET_FILE_MODULE = tpo.getenv_boolean("SET_FILE_MODULE", False)
DEFAULT_FILE_MODULE = "" if not SET_FILE_MODULE else os.path.splitext(os.path.basename(__file__))[0]
FILE_MODULE = tpo.getenv_text("FILE_MODULE", DEFAULT_FILE_MODULE).strip()

if USE_TRIE:
    import trie
if USE_SHELVE:
    import shelve
if USE_KYOTO:
    import kyotocabinet as kc

#------------------------------------------------------------------------

class TableLookup(object):
    """Abstract class for table lookup with support for incremental search"""
    __metaclass__ = ABCMeta

    # note: As workaround for pickling problem, explicitly encode full module path.
    if FILE_MODULE: __module__ = FILE_MODULE

    def __init__(self, filename=None, overwrite=True):
        """Class constructor"""
        # Note: overwrite applies to specializations requiring external data file
        # such as shelve or kyoto, in constrast to tries and hashes which need explcit save.
        tpo.debug_format("TableLookup.init([filename={f}, overwrite={ow}]); self=s", 7, 
                         f=filename, ow=overwrite, s=self)
        return

    @abstractmethod
    def dump(self, debug_level=7):
        """Traces out the object to stderr if DEBUG_LEVEL in effect"""
        return

    @abstractmethod
    def count(self):
        """Returns number of items in table"""
        return -1

    @abstractmethod
    def insert(self, words, value):
        """Insert key list of WORDS with VALUE into table"""
        return

    @abstractmethod
    def flush(self):
        """Flush data to file"""
        return

    @abstractmethod
    def lookup(self, words, context=None):
        """Return values for key list of WORDS, optionally relative to CONTEXT"""
        return

    @abstractmethod
    def has_prefix(self, words, context=None):
        """Whether table has key prefixed with list of WORDS, optionally relative to CONTEXT"""
        return

    def starts_with(self, word, context=None):
        """Whether table has key with prefix WORD, optionally relative to CONTEXT"""
        tpo.debug_print("Lookup.starts_with(%s, [%s])" % (word, context), 6)
        return(self.has_prefix([word], context))

    @abstractmethod
    def current_value(self, context=None):
        """Return current value relative to CONTEXT"""
        return
    
    @abstractmethod
    def keys(self):
        """Returns the list of keys in the table"""
        return []

    ## TODO:
    ## @abstractmethod
    ## def close(self):
    ##     """Close file handle"""
    ##     return

class TrieLookup(TableLookup):
    """Trie-based implementation of table lookup"""
    
    # note: As workaround for pickling problem, explicitly encode full module path.
    if FILE_MODULE: __module__ = FILE_MODULE

    def __init__(self, filename=None, overwrite=True):
        tpo.debug_format("TrieLookup.__init__([filename={f}, overwrite={ow}])\n", 6, 
                         f=filename, ow=overwrite)
        self.trie = trie.Trie(compressed=USE_PATRICIA_TRIE)
        return

    def dump(self, debug_level=7):
        """Traces out the object to stderr if DEBUG_LEVEL in effect"""
        tpo.debug_print(self.trie.format(), debug_level)
        return

    def count(self):
        """Returns number of items in table"""
        return self.trie.size()

    def insert(self, words, value):
        """Insert key list of WORDS with VALUE into table"""
        tpo.debug_print("TrieLookup.insert(%s, %s)" % (words, value), 6)
        if USE_CHAR_TRIE:
            words = " ".join(words)
        self.trie.insert(words, value)
        return

    def flush(self):
        """Flush data to file"""
        return

    def lookup(self, words, context=None):
        """Return value for key list of WORDS, optionally relative to CONTEXT (i.e., starting subtrie)"""
        ## Note: returns first value if more than one stored
        if USE_CHAR_TRIE:
            words = " ".join(words)
        start_trie = self.has_prefix(words, context)
        value = start_trie.current_value() if start_trie else None
        tpo.debug_print("TrieLookup.lookup(%s, [%s]) => %s" % (words, context, value), 5)
        return value

    def has_prefix(self, words, context=None):
        """Whether table has key prefixed with list of WORDS, optionally relative to CONTEXT (i.e., starting subtrie)"""
        if USE_CHAR_TRIE:
            words = " ".join(words)
        start_trie = context if context else self.trie
        if start_trie != self.trie:
            tpo.debug_print("start_trie: %s" % start_trie.format(), 9)
        # Find subnode corresponding to word sequence and make sure values at that node
        sub_trie = start_trie.find_prefix(words)
        tpo.debug_print("sub_trie: %s" % sub_trie and sub_trie.format(), 8)
        if (sub_trie and (len(sub_trie.get_all_values()) > 0)):
            tpo.debug_print("sub-trie values: %s; children-keys: %s" % (list(sub_trie.get_all_values()), sub_trie.children.keys()), 6)
        else:
            tpo.debug_print("TrieLookup.has_prefix: ignoring internal node (%s)" % sub_trie, 6)
            sub_trie = None
        tpo.debug_print("TrieLookup.has_prefix(%s, [%s]) => %s" % (words, context, sub_trie), 5)
        return (sub_trie)

    def current_value(self, context=None):
        """Return current value stored at CONTEXT (i.e., starting subtrie), using first if more than one"""
        start_trie = context if context else self.trie
        current_values = list(start_trie.get_all_values())
        value = current_values[0] if current_values else None
        return (value)

    def keys(self):
        """Returns the list of keys in the trie"""
        return self.trie.get_each_key()


class HashSlotLookup(TableLookup):
    """Hash-slot implementation of table lookup, inspired by GATE (see http://gate.ac.uk/sale/tao/splitch13.html). Their approach uses separates hashes for each possible prefix of subwords. That is supported as well as an expedient that just uses slot hashes to see whether any lookup entry has a word at a given position."""
    
    # note: As workaround for pickling problem, explicitly encode full module path.
    if FILE_MODULE: __module__ = FILE_MODULE

    def __init__(self, filename=None, overwrite=True, num_slots=10):
        "Initializes hash-slot lookup table with NUM_SLOTS of word-prefix hashes (0 for non-incremental lookup)"
        tpo.debug_format("HashSlotLookup.__init__([filename={f}, overwrite={ow}, [num_slots={ns})\n", 6,
                         f=filename, ow=overwrite, ns=num_slots)
        self.num_slots = num_slots
        self.subword_hash = []
        for _i in range(self.num_slots + 1):
            self.subword_hash.append({})
        ## TODO: self.remainder_hash = {}
        self.full_hash = {}
        gh.assertion(USE_SLOT_HASH or USE_PHRASE_SLOTS)
        return

    def dump(self, debug_level=7):
        tpo.debug_print("HashSlotLookup: {", debug_level)
        tpo.debug_print("    num_slots=%s" % self.num_slots, debug_level)
        for i in range(self.num_slots):
            tpo.debug_print("    subword_hash[%d]=%s" % (i, self.subword_hash[i]), debug_level)
        ## TODO: tpo.debug_print("    remainder_hash=%s" % self.remainder_hash, debug_level)
        tpo.debug_print("    full_hash=%s" % self.full_hash, debug_level)
        tpo.debug_print("    }", debug_level)
        return

    def count(self):
        return len(self.full_hash)

    def insert(self, words, value):
        tpo.debug_print("HashSlotLookup.insert(%s, %s)" % (words, value), 6)
        # Insert complete phrase into full hash
        phrase = " ".join(words)
        self.full_hash[phrase] = value
        # Set indicator hashes for each subword
        num_key_slots = min(len(words), self.num_slots)
        for i in range(num_key_slots):
            key = words[i] if USE_WORD_SLOTS else " ".join(words[:(i + 1)])
            self.subword_hash[i][key] = True
        # Add remaining subphrase to catchall hash
        ## TODO: if (num_key_slots < len(words)):
            ## TODO: self.remainder_hash[" ".join(words[num_key_slots:])] = True
        if (num_key_slots < len(words)):
            tpo.debug_print("Warning: too many words for slot-hash lookup: %s" % words)
        return

    def flush(self):
        """Flush data to file"""
        return

    def lookup(self, words, context=None):
        assert(not context)
        value = None
        phrase = " ".join(words)
        if phrase in self.full_hash:
            value = self.full_hash[phrase]
        tpo.debug_print("HashSlotLookup.lookup(%s) => %s" % (words, value), 5)
        return (value)

    def has_prefix(self, words, context=None):
        """Whether table has key prefixed with list of WORDS, optionally relative to CONTEXT (e.g., starting slot)"""
        found = True

        # Special case of phrase slots (approach used in GATE)
        # note: context is list of previous words found (not slot number as usual)
        if USE_PHRASE_SLOTS:
            new_context = context + words if context else words
            if (len(words) < self.num_slots):
                key = " ".join(new_context)
                slot = len(new_context) - 1
                if (key not in self.subword_hash[slot]):
                    tpo.debug_print("key '%s' not in slot %d" % (key, slot), 6)
                    new_context = None
            tpo.debug_print("HashSlotLookup.has_prefix(%s, [%s]) => %s" % (words, context, new_context), 5)
            return (new_context)

        # Check each word for occurrence in slot-specific hash
        start_slot = context if context else 0
        last_slot_plus = min(start_slot + len(words), self.num_slots)
        i = 0
        while ((start_slot + i) < last_slot_plus):
            if (words[i] not in self.subword_hash[start_slot + i]):
                found = False
                tpo.debug_print("Word '%s' not found in slot %d hash" % (words[i], i), 6)
                break
            i += 1

        # TODD: Check remaining words in catchall hash
#        if (found) and ((start_slot + i) == self.num_slots) and (i < len(words)):
#            remainder = " ".join(words[i:])
#            if remainder not in self.remainder_hash:
#                found = False
#                tpo.debug_print("Remainder '%s' not found in remainder hash" % remainder, 6)
        if (found) and ((start_slot + i) == self.num_slots) and (i < len(words)):
            found = False

        # Returns index of next slot if found, otherwise null
        new_context = (start_slot + i) if found else None
        tpo.debug_print("HashSlotLookup.has_prefix(%s, [%s]) => %s" % (words, context, new_context), 5)
        return new_context

    def current_value(self, context=None):
        """Current value not applicable for slot-hash lookup"""
        return None

    def keys(self):
        return self.full_hash.keys()


class HashDbLookup(TableLookup):
    """Table lookup via hash-like object cached to disk"""
    # Note: only supports brute-force lookup (i.e., non-incremental)

    def __init__(self, filename="table_lookup.hash-db.data", overwrite=True):
        tpo.debug_format("HashDbLookup.__init__([filename={f}, overwrite={ow}])\n", 6,
                         ow=overwrite, f=filename)
        self.filename = filename
        return

    def has_prefix(self, words, context=None):
        """Whether table has key prefixed with list of WORDS, optionally relative to CONTEXT"""
        tpo.print_stderr("has_prefix not supported for hash-db tables")
        return None

    def starts_with(self, word, context=None):
        """Whether table has key with prefix WORD, optionally relative to CONTEXT"""
        tpo.print_stderr("starts_with not supported for hash-db tables")
        return None

    def current_value(self, context=None):
        """Return current value relative to CONTEXT"""
        tpo.print_stderr("current_value not supported for hash-db tables")
        return None

    @classmethod
    def from_hash(cls, hash_filename, hash_db_filename):
        """Converts hash-based look into db-hash
        Note: currently just intended for interactive use"""
        # TODO: rework to use serialized hash
        hash_db = HashDbLookup(hash_db_filename)
        # Note: uses create_lookup_table, which is just for hashes
        hash_table = tpo.create_lookup_table(hash_filename)
        for (k, v) in enumerate(hash_table):
            hash_db.insert(k, v)
        return hash_db

class ShelveLookup(HashDbLookup):
    """Table lookup via python shelve (db-backed hash)"""

    # note: As workaround for pickling problem, explicitly encode full module path.
    if FILE_MODULE: __module__ = FILE_MODULE

    def __init__(self, filename="table_lookup.shelve.data", overwrite=True):
        tpo.debug_format("ShelveLookup.__init__([filename={f}, overwrite={ow}])\n", 6,
                         ow=overwrite, f=filename)
        if overwrite and os.path.exists(filename):
            gh.delete_file(filename)
        if verbose:
            action = "Loading" if os.path.exists(filename) else "Saving"
            print("{act} table {f}".format(act=action, f=filename))
        self.data_store = shelve.open(filename)
        return

    def dump(self, debug_level=7):
        """Traces out the object to stderr if DEBUG_LEVEL in effect (and __debug__)"""
        tpo.debug_print("ShelveLookup: data_store=%s" % self.data_store, debug_level)
        return

    def count(self):
        """Returns number of items in table"""
        # Note: not supported as this requires bringing all of the stored data into memory
        return -1

    def insert(self, words, value):
        """Insert key list of WORDS with VALUE into table"""
        tpo.debug_print("ShelveLookup.insert(%s, %s)" % (words, value), 6)
        phrase = " ".join(words)
        self.data_store[phrase] = value
        return

    def flush(self):
        """Flush data to file"""
        return

    def lookup(self, words, context=None):
        """Return values for key list of WORDS, optionally relative to CONTEXT"""
        phrase = " ".join(words)
        value = self.data_store[phrase] if (phrase in self.data_store) else None
        tpo.debug_print("ShelveLookup.lookup(%s) => %s" % (words, value), 5)
        return (value)

    def keys(self):
        """Returns list of keys in the table"""
        return self.data_store.keys()

class KyotoLookup(HashDbLookup):
    """Table lookup via python kyoto (db-backed hash)"""
    KCT = "kct"
    KCT_EXT = "." + KCT
    if FILE_MODULE: __module__ = FILE_MODULE

    def __init__(self, filename=None, overwrite=True):
        """Class constructor: includes symbolic link hack for embedded period issue in filename"""
        tpo.debug_format("KyotoLookup.__init__([filename={f}, overwrite={ow}])\n", 6,
                         ow=overwrite, f=filename)
        if not filename:
            filename = "table_lookup_kyoto" + self.KCT_EXT
        self.data_store = kc.DB()
        file_mode = None
        # Normalize the filename for use with quirky Kyoto
        ## HACK: ensure no embedded periods (and use symbolic link for user name)
        internal_filename = filename
        if not internal_filename.endswith(self.KCT_EXT):
            internal_filename += self.KCT_EXT
        internal_filename = internal_filename.replace(".", "_")
        internal_filename = internal_filename.replace("_" + self.KCT, self.KCT_EXT)
        tpo.debug_format("internal_filename={f}", 5, f=internal_filename)
        gh.assertion(internal_filename.endswith(self.KCT_EXT))
        # Determine the file mode
        file_mode = None
        if overwrite:
            file_mode = (kc.DB.OWRITER | kc.DB.OCREATE)
            if os.path.exists(internal_filename):
                gh.delete_file(internal_filename)
            if verbose:
                print("Saving table {f}".format(f=internal_filename))
        else:
            if verbose:
                print("Loading table {f}".format(f=internal_filename))
            file_mode = kc.DB.OREADER
        # Open the file and create symbolic link to user name
        open_ok = self.open(internal_filename, file_mode)
        gh.assertion(open_ok)
        if overwrite and (internal_filename != filename):
            tpo.debug_format("Warning: Creating symbolic link to internal file for Kyoto", 3,
                             int=internal_filename)
            gh.run("ln -fs {int} {f}", f=filename, int=internal_filename)
        return

    def open(self, filename, mode=None):
        """Open FILENAME in MODE"""
        if mode is None:
            mode = kc.DB.OREADER
        ok = self.data_store.open(filename, mode)
        tpo.debug_format("KyotoLookup.open({f}, {m}) => {r}; self={s}", 6, 
                         f=filename, m=mode, r=ok, s=self)
        return ok

    def close(self):
        """Open FILENAME in MODE"""
        tpo.debug_format("KyotoLookup.close(); self={s}", 6, s=self)
        return self.data_store.close()

    def dump(self, debug_level=7):
        """Traces out the object to stderr if DEBUG_LEVEL in effect (and __debug__)"""
        tpo.debug_print("KyotoLookup: data_store=%s" % self.data_store, debug_level)
        # TODO: use iterate method
        #       dump= []; db.iterate(lambda k,v: dump.append((k, v)))'
        # see https://github.com/KosyanMedia/kyotocabinet-python/blob/master/kyotocabinet-doc.py
        return

    def count(self):
        """Returns number of items in table"""
        num = len(self.data_store)
        tpo.debug_format("KyotoLookup.coount() => {n}", 6, n=num)
        return num

    def insert(self, words, value):
        """Insert key list of WORDS with VALUE into table"""
        tpo.debug_print("KyotoLookup.insert(%s, %s)" % (words, value), 6)
        phrase = " ".join(words)
        insert_ok = self.data_store.add(phrase, value)
        gh.assertion(insert_ok)
        # TODO: return insert_ok
        return

    def lookup(self, words, context=None):
        """Return values for key list of WORDS, optionally relative to CONTEXT"""
        phrase = " ".join(words)
        value = self.data_store.get(phrase)
        tpo.debug_print("KyotoLookup.lookup(%s) => %s" % (words, value), 5)
        return (value)

    def flush(self):
        """Flush data to file"""
        tpo.debug_format("KyotoLookup.flush(); self={s}", 6, s=self)
        self.data_store.synchronize()
        return

    def keys(self):
        """Returns the list of keys in the table"""
        key_names = []
        def lookup(key, _value):
            """Helper for kytooto iterate"""
            key_names.append(key)
        self.data_store.iterate(lookup)
        return key_names

#------------------------------------------------------------------------

def read_lookup_table(table, filename):
    """Populate TABLE with entries from FILENAME.
    Input Format: Key[<tab><Value>]"""
    f = None
    try:
        f = open(filename)
        for line_num, line in enumerate(f):
            ## OLD: line = line.strip().lower()
            tpo.debug_print("table L%d: %s" % (line_num + 1, line), 7)

            # Check for phrase with optional tab-separated value
            match = re.match("^([^\t]*)(\t(.*))?", line)
            if match:
                ## OLD: key = match.group(1)
                key = match.group(1).strip().lower()
                value = match.group(3) if match.group(3) else True
                if (line_num == 0):
                    # Add special entry to indicate label for value
                    label = value if (value != True) else "Value"
                    table.insert([""], label)
                words = re.split(r"\s+", key)

                # Add entry unless exceeds length limit
                if (len(words) <= MAX_PHRASE_LEN):
                    table.insert(words, value)
                else:
                    tpo.print_stderr("Max phrase length ({max}) exceeded in key ({len}): {phr}",
                                     max=MAX_PHRASE_LEN, key=len(words), phr=words)
            else:
                tpo.debug_print("Ignoring line %d of %s: %s" % (line_num + 1, line, filename))
    except (IOError, ValueError):
        tpo.debug_print("Warning: Exception reading lookup table from %s: %s" % (filename, str(sys.exc_info())), 2)
    finally:
        if f:
            f.close()
    return


def open_lookup_table(filename):
    """Open FILENAME of type secified by environemt ooptions lie USE_KYOTO"""
    if USE_SHELVE:
        lookup_class = ShelveLookup
    elif USE_KYOTO:
        lookup_class = KyotoLookup
    elif USE_TRIE:
        lookup_class = TrieLookup
    else:
        lookup_class = HashSlotLookup
    table = lookup_class(overwrite=False, filename=filename)
    tpo.debug_format("open_lookup_table({f}) => {t}", 4, f=filename, t=table)
    return table


def create_serialized_lookup_table(input_filename, save_filename=None, load_filename=None):
    """Create lookup table from term mappings in INPUT_FILENAME (one pair per line), using lowercase keys. The data is object written to SAVE_FILENAME (e.g., pickle format). If the input filename is -, then the table is loaded from LOAD_FILENAME"""
    # Note: for transparent handling of shelve-based tables, the input filename can be specified as - for n/a
    tpo.debug_print("create_serialized_lookup_table(%s, [save_filename=%s], [load_filename=%s])" % (input_filename, save_filename, load_filename), 4)
    if USE_SHELVE:
        table = ShelveLookup(save_filename or load_filename, overwrite=save_filename)
    elif USE_KYOTO:
        table = KyotoLookup(save_filename or load_filename, overwrite=save_filename)
        ## TODO: table.flush()
        ## TEST: table.close(); table.open(save_filename or load_filename)
    else:
        table = TrieLookup() if USE_TRIE else HashSlotLookup()
    if (input_filename != "-"):
        read_lookup_table(table, input_filename)
        tpo.trace_object(table, 9, "table")
        if save_filename and not USE_DB_HASH:
            if verbose:
                print("Saving table {f}".format(f=save_filename))
            tpo.store_object(save_filename, table)
    else:
        if load_filename and not USE_DB_HASH:
            if verbose:
                print("Loading table {f}".format(f=load_filename))
            table = tpo.load_object(load_filename)
    table.dump()
    return table


def verify_table_lookup(table):
    """Verifies that each entry from STDIN can be successfully retrieved from TABLE"""
    # Search for items from table occurring in input
    tpo.debug_print("Verifing table lookup", tpo.ALWAYS)
    num_found = 0
    line_num = 0
    for line in sys.stdin.readlines():
        line = line.strip()
        line_num += 1
        ## TODO: debug_print("L%d: %s" % (fileinput.filelineno(), line), 5)
        tpo.debug_print("L%d: %s" % (line_num, line), 5)
        terms = re.split(r"\s+", line.lower())

        start = 0
        found_info = []
        while (start < len(terms)):
            phrase_len = 1

            # See if a known phrase starts with the current word.
            # If using brute-force lookup, assume all of remainder matches.
            context = None
            if BRUTE_FORCE:
                phrase_len = len(terms) - start
            else:
                # Start incremental search if current word starts a phrase
                context = table.starts_with(terms[start], None)
                if (context):
                    # Continue adding words while current one might continue a known phrase
                    while (context and ((start + phrase_len) < len(terms))):
                        context = table.starts_with(terms[start + phrase_len], context)
                        if context:
                            phrase_len += 1
                else:
                    phrase_len = 0
                
            # Verify maximal phrase as lookahead might be heuristic
            while (phrase_len > 0):
                ## OLD: subphrase = " ".join(terms[start:(start + phrase_len)])
                words = terms[start:(start + phrase_len)]
                value = None
                if context:
                    value = table.current_value(context)
                if not value:
                    value = table.lookup(words)
                if value:
                    phrase = " ".join(words)
                    info = phrase if value is True else (phrase, value)
                    found_info.append(info)
                    num_found += 1
                    break
                phrase_len -= 1

            start += max(1, phrase_len)

        # Output matching subphrases
        print("%s => %s" % (line, found_info))

    # Display some statistics on the lookup results
    print("%d subphrases from %d input lines found" % (num_found, line_num))
            

def main():
    """Entry point for script"""
    # Parse arguments: table-data search-data
    env_options = tpo.formatted_environment_option_descriptions()
    notes = """
Notes:
- In most cases, external data tables must explicitly be created via --save.
- The following environment options are available:
        {env_opts}
""".format(env_opts=env_options)
     
    parser = argparse.ArgumentParser(description=__doc__, epilog=notes,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    ## TODO: use lookup-file
    parser.add_argument("lookup_file", help="file with keys and values (e.g., named entities and weights)")
    ## TEST: parser.add_argument("lookup-file", help="file with known entities and weight")
    parser.add_argument("--save-file", help="file for storing lookup-table instance")
    parser.add_argument("--load-file", help="file for loading lookup-table instance")
    parser.add_argument("--verbose", default=False, action='store_true', help="Verbose output mode")
    parser.add_argument("--skip-test", default=False, action='store_true', help="skip the verification search for each item in table")
    parser.add_argument("--print-table", default=False, action='store_true', help="print out the table in plain text format")
    ## OLD: parser.add_argument("search_file", help="file with new cases to lookup")
    # TODO: show environment options
    args = parser.parse_args()
    tpo.debug_print("args: %s" % args, 4)
    global verbose
    verbose = args.verbose

    # Create (or load) slot-hash or trie for incremental lookup (or shelve for exact lookup)
    table = create_serialized_lookup_table(args.lookup_file, args.save_file, args.load_file)

    if (not args.skip_test):
        verify_table_lookup(table)
    if (args.print_table):
        for k in table.keys():
            print("%s\t%s" % (k, table.lookup(k)))


    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
