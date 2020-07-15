#! /usr/bin/env python
#
# Implementation of trie data structure based on Ruby version by Daniel Erat:
#    Copyright 2005 Daniel Erat <dan-ruby@erat.org>: GNU GPL (see erat-trie-COPYING.txt)
# via https://github.com/dustin/ruby-trie/blob/master/lib/trie.rb
#
# TODO:
# - convert iterators from Ruby original: each, each_key, each_value
# - likewise convert custom serializaion code (_dump and _load)
#

import os
import re
import sys
from tpo_common import debug_print, getenv_integer, print_stderr

MAX_TRIE_KEY_LEN = getenv_integer("MAX_TRIE_KEY_LEN", 25)

#-------------------------------------------------------------------------

class Trie:
    """Trie data structure with optional node compression (as in a Patricia Trie). This support both strings (i.e., character trie) and sequences (e.g., string trie)."""
    make_compressed = False

    def __init__(self, compressed=None):
        """Creates a new empty trie, optionally with node compression"""
        # Note: Maintains null compressed key to facilitate comparisons below
        # TODO: Overload compressed_key?
        debug_print("Trie.__init__(compressed=%s)" % compressed, 6)
        if not compressed:
            compressed = Trie.make_compressed
        self.values = set()
        self.children = {}
        self.is_compressed = compressed
        self.compressed_key = None
        self.compressed_values = None
        if self.is_compressed:
            self.compressed_key = []
            self.compressed_values = set()
        return

    def __getitem__(self, key):
        """Support for [] operator: returns all of the items matching a key"""
        debug_print("Trie.__getitem__(%s)" % key, 6)
        return (self.find(key).values)

    def clear(self):
        """Clear the trie"""
        debug_print("Trie.clear()", 6)
        self.values.clear()
        self.children.clear()
        if self.is_compressed:
           self.compressed_key = []
           self.compressed_values.clear()
        return

    def delete(self, key):
        """Delete all values with a given key"""
        debug_print("Trie.delete(%s)" % key, 6)
        if not key:
            self.values.clear()
        elif (key == self.compressed_key):
            self.compressed_key = []
            self.compressed_values.clear()
        elif key[0] in self.children:
            self.children[key[0]].delete(key[1:])
            if self.children[key[0]].is_empty():
                del(self.children[key[0]])
        return (self)

    def delete_value(self, value):
        """Delete all occurences of an value"""
        debug_print("Trie.delete_value(%s)" % value, 6)
        if self.is_compressed:
            if (value in self.compressed_values):
                self.compressed_values.remove(value)
            if not self.compressed_values:
                self.compressed_key = ""
        if (value in self.values):
            self.values.remove(value)
        for (k, t) in self.children.items():
            t.delete_value(value)
            if t.is_empty():
                del(self.children[k])
        return (self)

    def delete_pair(self, key, value):
        """Delete a (key, value) pair"""
        debug_print("Trie.delete_pair(%s, %s)" % (key, value), 6)
        if not key:
            if (value in self.values):
                self.values.remove(value)
        elif (key == self.compressed_key):
            if (value in self.compressed_values):
                self.compressed_values.remove(value)
            self.compressed_key.clear()
        elif key[0] in self.children:
            self.children[key[0]].delete_pair(key[1:], value)
            if (self.children[key[0]].is_empty()):
                del(self.children[key[0]])
        return (self)

    def delete_prefix(self, prefix):
        """Delete all values keyed by a given prefix"""
        debug_print("Trie.delete_prefix(%s)" % prefix, 6)
        if (not prefix) or (self.is_compressed and (prefix == self.compressed_key[0:len(prefix)])):
            self.clear()
        elif prefix[0] in self.children:
            self.children[prefix[0]].delete_prefix(prefix[1:])
            if self.children[prefix[0]].is_empty():
                del(self.children[prefix[0]])
        return (self)
  
    def is_empty(self):
        """Whether the Trie contains no values"""
        debug_print("Trie.is_empty()", 6)
        return (self.size() == 0)
  
    def find(self, key):
        """Get a new Trie object containing all values with the passed-in key"""
        debug_print("Trie.find(%s); self=%s" % (key, self), 6)
        new_trie = None
        if ((not key) and (not self.compressed_key)) or (key == self.compressed_key):
            new_trie = Trie(self.is_compressed)
            for v in self.values:
                new_trie.insert([], v)
            if self.compressed_values:
                for v in self.compressed_values:
                    new_trie.insert([], v)
        elif (len(key) > 0) and (key[0] in self.children):
            new_trie = self.children[key[0]].find(key[1:])
        else:
            new_trie = Trie(self.is_compressed)
        return (new_trie)
  
    def find_prefix(self, prefix):
        """Return a Trie object containing all values with keys that begin with the passed-in prefix"""
        debug_print("Trie.find_prefix(%s); self=%s" % (prefix, self), 6)
        new_trie = None
        if (not prefix):
            # TODO: make copy?
            new_trie = self
        elif self.is_compressed and (prefix == self.compressed_key[0:len(prefix)]):
            new_trie = Trie(self.is_compressed)
            for value in self.compressed_values:
                new_trie.insert(self.compressed_key[len(prefix):], value)
        elif prefix[0] in self.children:
            new_trie = self.children[prefix[0]].find_prefix(prefix[1:])
        else:
            new_trie = Trie(self.is_compressed)
        return (new_trie)
  
    def insert(self, key, value):
        """Insert an value into this Trie, keyed by the passed-in key, which can be any sequence object."""
        debug_print("Trie.insert(%s, %s); self=%s" % (key, value, self), 6)
        if (len(key) > MAX_TRIE_KEY_LEN):
            print_stderr("Max key length (%d) exceeded in key (%d): %s" % (MAX_TRIE_KEY_LEN, len(key), key))
            return None
        if self.is_compressed and (key != self.compressed_key):
            for v in self.compressed_values:
                self._insert_in_child(self.compressed_key, v)
            self.compressed_values.clear()
            self.compressed_key = []
        if not key:
            self.values.add(value)
        # TODO: simply test (use separate function for compressed case?)
        elif self.is_compressed and (((not self.values) and (not self.children)) or (key == self.compressed_key)):
            self.compressed_key = key[:]
            self.compressed_values.add(value)
        else:
            self._insert_in_child(key, value)
        return (self)
  
    def _insert_in_child(self, key, value):
        """Insert an value into a sub-Trie, creating one if necessary."""
        debug_print("Trie._insert_in_child(%s, %s); self=%s" % (key, value, self), 6)
        # note: Internal method called by Trie.insert.
        if key[0] not in self.children:
            self.children[key[0]] = Trie(self.is_compressed)
        return (self.children[key[0]].insert(key[1:], value))
  
    def get_each_key(self, prefix=[]):
      """Returns all keys in the trie (optionally with given prefix added)"""
      debug_print("Trie.get_each_key(%s)" % prefix, 6)
      # Note: approximation to each_key [iterator]
      all_keys = []
      if self.values:
          all_keys += prefix[:]
      if (self.compressed_values):
          all_keys += self.compressed_key[:]
      for k in self.children:
          all_keys += self.children[k].all_keys(prefix[:].append(k))
      return (all_keys)
  
    def get_each_value(self):
        """Returns all values in the trie"""
        debug_print("Trie.get_each_value()", 6)
        # TODO: make sure mutable values copied
        all_values = []
        if self.compressed_values:
            all_values += self.compressed_values.copy()
        if self.values:
            all_values += self.values.copy()
        for k in self.children:
            all_values += self.children[k].get_each_value()
        return (all_values)
  
    def keys(self):
        """Get an array containing all keys in this Trie"""
        debug_print("Trie.keys()", 6)
        return(self.all_keys())
  
    def num_nodes(self):
        """Get the number of nodes used to represent this Trie."""
        debug_print("Trie.num_nodes()", 6)
        # This is only useful for testing.
        node_count = 1
        for k in self.children:
            node_count += self.children[k].num_nodes()
        return (node_count)
  
    def size(self):
        """Get the number of values contained in this Trie"""
        debug_print("Trie.size()", 6)
        total_size = len(self.values)
        for k in self.children:
            total_size += self.children[k].size()
        if self.compressed_values:
            total_size += len(self.compressed_values)
        return (total_size)
  
    def get_all_values(self):
        """Get an Array containing all values in this trie"""
        debug_print("Trie.get_all_values()", 6)
        return(self.get_each_value())

    def format(self, level=1):
        """Return formatted version of TRIE"""
        debug_print("Trie.format(%d)" % level, 6)
        indent = " " * level
        result = """trie: values={VALUES}; comp?={IS_COMPRESSED}; comp_key={COMPRESSED_KEY}; comp_values={COMPRESSED_VALUES}; children:"""
        result = result.format(VALUES=self.values, IS_COMPRESSED=self.is_compressed, COMPRESSED_KEY=self.compressed_key, COMPRESSED_VALUES=self.compressed_values, INDENT=indent)
        for (key, subtrie) in self.children.items():
            result += "\n" + indent + "'%s': %s" % (key, subtrie.format(level +1))
        return(result)

#------------------------------------------------------------------------

def simple_test():
    """Runs simple test of trie (see test-trie.py for detailed unit tests)"""
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    compressed = (len(sys.argv) > 1) and (sys.argv[1] == "--compressed")
    ## t = Trie(compressed=compressed)
    t = Trie(compressed)
    print("t = %s" % t.format())
    t.insert("the", 1)
    print("t = %s" % t.format())
    t.insert("they", 2)
    t.insert("they", 3)
    t.insert("their", 4).insert("they're", 5)
    print("t = %s" % t.format())
    print("expected values: [1, 2, 3, 4, 5]")
    print("t values: %s" % sorted(t.get_all_values()))
    print("")
    
    print('Search for an exact match of "they":')
    print("expected values: [2, 3]")
    print("values: %s" % sorted(t.find("they").get_all_values()))
    print("")

    print('Search for prefix "th" that matches all keys.')
    t2 = t.find_prefix("th")
    print("t2 = %s" % t.format())
    print("expected size: 5")
    print("t2 size: %d" % t2.size())
    print("expected values: [1, 2, 3, 4, 5]")
    print("t2 values: %s" % sorted(t2.get_all_values()))
    print("")

    print('In the sub-Trie beginning with "th", search for the prefix "ey"')
    # returns the three values with keys beginning with "they".
    print("expected values: [2, 3, 5]")
    print("result: %s" % sorted(t2.find_prefix("ey").get_each_value()))
    print("")

    print('Now searching for "at" in the sub-Trie, which should results in an empty Trie')
    # note: there are no keys beginning with "that"
    print("result: %s" % t2.find_prefix("at").get_each_value())
    print("result empty?: %s" % t2.find_prefix("at").is_empty())
    print("")

    print('Delete all values keyed by "they"')
    # Note: For compressed tries, this must be performed on the root Trie rather than the one
    # returned by find_prefix(): see Notes under https://github.com/dustin/ruby-trie/blob/master/lib/trie.rb
    t3 = t.delete("they")
    print("t3 = %s" % t3.format())
    print("expected values: [1, 4, 5]")
    print("t3 values: %s" % sorted(t3.get_all_values()))
    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    print_stderr("Warning: %s is not intended to be run standalone" % __file__)
    print_stderr("Running simple test")
    simple_test()
