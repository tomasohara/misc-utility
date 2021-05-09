#! /usr/bin/env python
#
# Test suite for implementation of trie data structure based on Ruby version by Daniel Erat:
# Copyright 2005 Daniel Erat <dan-ruby@erat.org>: GNU GPL (see erat-trie-COPYING.txt)
# via https://github.com/dustin/ruby-trie/blob/master/tests/trie.rb
#

import tpo_common as tpo
import unittest
from trie import *

MAKE_COMPRESSED = tpo.getenv_boolean("MAKE_COMPRESSED", False)

class TestTrie(unittest.TestCase):
    """Unit tests for the Trie class."""

    def test_find_compressed_key_single_value_at_root(self):
        """Test a compressed key with a single value at the root"""
        t = Trie(MAKE_COMPRESSED).insert('abc', 1)
        self.assertEqual([1], t.find('abc').get_all_values())
        self.assertEqual([1], t.find('abc').find('').get_all_values())
        self.assertEqual([], t.find('a').get_all_values())
        self.assertEqual([], t.find('').get_all_values())
        self.assertEqual([], t.find('b').get_all_values())
        self.assertEqual([1], t.find_prefix('abc').get_all_values())
        self.assertEqual([1], t.find_prefix('abc').find_prefix('').get_all_values())
        self.assertEqual([1], t.find_prefix('ab').get_all_values())
        self.assertEqual([1], t.find_prefix('a').get_all_values())
        self.assertEqual([1], t.find_prefix('').get_all_values())
        self.assertEqual([], t.find_prefix('b').get_all_values())
        return

    def test_find_compressed_key_multiple_values_at_root(self):
        """Test a compressed key with multiple values at the root"""
        t = Trie(MAKE_COMPRESSED).insert('ab', 1).insert('ab', 2).insert('ab', 3)
        self.assertEqual([1, 2, 3], sorted(t.find('ab').get_all_values()))
        self.assertEqual([], t.find('a').get_all_values())
        self.assertEqual([], t.find('').get_all_values())
        self.assertEqual([1, 2, 3], sorted(t.find_prefix('ab').get_all_values()))
        self.assertEqual([1, 2, 3], sorted(t.find_prefix('a').get_all_values()))
        self.assertEqual([1, 2, 3], sorted(t.find_prefix('').get_all_values()))
        return

    def test_find_complex(self):
        """Test a more complex Trie that contains a few compressed keys"""
        t = Trie(MAKE_COMPRESSED).insert('a', 1).insert('ab', 2).insert('abcdef', 3).insert('b', 4).insert('bcd', 5).insert('b', 6).insert('bcd', 7)
        self.assertEqual([1], t.find('a').get_all_values())
        self.assertEqual([2], t.find('ab').get_all_values())
        self.assertEqual([3], t.find('abcdef').get_all_values())
        self.assertEqual([4, 6], sorted(t.find('b').get_all_values()))
        self.assertEqual([5, 7], sorted(t.find('bcd').get_all_values()))
        self.assertEqual([], t.find('bcde').get_all_values())
        self.assertEqual([], t.find('').get_all_values())
        self.assertEqual([1, 2, 3], t.find_prefix('a').get_all_values())
        self.assertEqual([2, 3], t.find_prefix('ab').get_all_values())
        self.assertEqual([3], t.find_prefix('abcdef').get_all_values())
        self.assertEqual([4, 5, 6, 7], sorted(t.find_prefix('b').get_all_values()))
        self.assertEqual([5, 7], sorted(t.find_prefix('bcd').get_all_values()))
        self.assertEqual([], t.find_prefix('bcde').get_all_values())
        self.assertEqual([1, 2, 3, 4, 5, 6, 7], sorted(t.find_prefix('').get_all_values()))
        return

    def test_find_multiple_lookups_compressed_key(self):
        """Creat a compressed key at the root and then do one-or two-characters-at-a-time searches against it."""
        t = Trie(MAKE_COMPRESSED).insert('alphabet', 1)
        t2 = t.find_prefix('')
        self.assertEqual([1], t2.get_all_values())
        t2 = t2.find_prefix('al')
        self.assertEqual([1], t2.get_all_values())
        t2 = t2.find_prefix('p')
        self.assertEqual([1], t2.get_all_values())
        t2 = t2.find_prefix('ha')
        self.assertEqual([1], t2.get_all_values())
        t2 = t2.find_prefix('bet')
        self.assertEqual([1], t2.get_all_values())
        t2 = t2.find_prefix('')
        self.assertEqual([1], t2.get_all_values())
        t2 = t2.find_prefix('a')
        self.assertEqual([], t2.get_all_values())
        return

    def test_find_multiple_lookups(self):
        """Constructs a trie with multiple values and then walks down it, searching for one or two characters at a time"""
        t = Trie(MAKE_COMPRESSED).insert('happy', 1).insert('hop', 2).insert('hey', 3).insert('hello!', 4).insert('help', 5).insert('foo', 6)
        self.assertEqual([6], t.find_prefix('fo').get_all_values())
        t2 = t.find_prefix('h')
        self.assertEqual([1, 2, 3, 4, 5], sorted(t2.get_all_values()))
        t2 = t2.find_prefix('e')
        self.assertEqual([3, 4, 5], sorted(t2.get_all_values()))
        self.assertEqual([3], t2.find_prefix('y').get_all_values())
        t2 = t2.find_prefix('l')
        self.assertEqual([4, 5], sorted(t2.get_all_values()))
        t2 = t2.find_prefix('lo')
        self.assertEqual([4], t2.get_all_values())
        t2 = t2.find_prefix('!')
        self.assertEqual([4], t2.get_all_values())
        t2 = t2.find_prefix('')
        self.assertEqual([4], t2.get_all_values())
        t2 = t2.find_prefix('!')
        self.assertEqual([], t2.get_all_values())
        return

    def test_size(self):
        """Construct a trie with multiple elements and test the size"""
        t = Trie(MAKE_COMPRESSED).insert('ha', 1).insert('hat', 2).insert('hate', 3).insert('hated', 4).insert('test', 5)
        self.assertEqual(5, t.size())
        self.assertEqual(4, t.find_prefix('ha').size())
        self.assertEqual(2, t.find_prefix('hate').size())
        self.assertEqual(1, t.find_prefix('test').size())
        self.assertEqual(0, t.find_prefix('testing').size())
        return

    def test_empty(self):
        """Build a trie and test the empty? method."""
        t = Trie(MAKE_COMPRESSED).insert('foo', 1).insert('bar', 2).insert('food', 3)
        self.assertEqual(False, t.is_empty())
        self.assertEqual(False, t.find('foo').is_empty())
        self.assertEqual(False, t.find_prefix('foo').is_empty())
        self.assertEqual(True, t.find('fool').is_empty())
        return

    def test_mixed_classes_in_keys(self):
        """Insert keys that are actually lists containing objects of varying classes"""
        t = Trie(MAKE_COMPRESSED).insert([0, 1, 2], 0).insert([0, 'a'], 1).insert([1000], 2).insert([0, 'a'], 3).insert('blah', 4)
        self.assertEqual([0, 1, 3], sorted(t.find_prefix([0]).get_all_values()))
        self.assertEqual([0], t.find_prefix([0, 1]).get_all_values())
        self.assertEqual([1, 3], sorted(t.find_prefix([0, 'a']).get_all_values()))
        self.assertEqual([2], t.find_prefix([1000]).get_all_values())
        self.assertEqual([], t.find([0]).get_all_values())
        self.assertEqual([1, 3], sorted(t.find([0, 'a']).get_all_values()))
        self.assertEqual([4], t.find('blah').get_all_values())
        return

    def test_delete(self):
        """Test delete"""
        t = Trie(MAKE_COMPRESSED).insert('a', 1).insert('a', 2).insert('a', 3).insert('ab', 4).insert('ab', 5).insert('abc', 6)
        self.assertEqual([1, 2, 3, 4, 5, 6], sorted(t.get_all_values()))
        t.delete('a')
        self.assertEqual([4, 5, 6], sorted(t.get_all_values()))
        t.delete('abc')
        self.assertEqual([4, 5], sorted(t.get_all_values()))
        t.delete('ab')
        self.assertEqual([], t.get_all_values())
        return

    def test_delete_pair(self):
        """Tests delete_pair"""
        t = Trie(MAKE_COMPRESSED).insert('apple', 1).insert('apples', 2)
        self.assertEqual([1], t.find('apple').get_all_values())
        self.assertEqual([1, 2], sorted(t.find_prefix('apple').get_all_values()))
        t.delete_pair('apple', 1)
        self.assertEqual([], t.find('apple').get_all_values())
        self.assertEqual([2], t.find('apples').get_all_values())
        self.assertEqual([2], t.find_prefix('apple').get_all_values())
        t.delete_pair('apples', 1)  # key/value pair isn't in trie
        self.assertEqual([2], t.find('apples').get_all_values())
        t.delete_pair('apples', 2)
        self.assertEqual([], t.find('apples').get_all_values())
        return

    def test_delete_value(self):
        """Tests delete_value"""
        t = Trie(MAKE_COMPRESSED).insert('a', 1).insert('ab', 1).insert('abc', 2).insert('a', 2).insert('b', 1).insert('c', 1)
        self.assertEqual(6, t.size())
        t.delete_value(1)
        self.assertEqual(2, t.size())
        t.delete_value(2)
        self.assertEqual(True, t.is_empty())
        return

    def test_delete_prefix(self):
        """Tests delete_prefix"""
        t = Trie(MAKE_COMPRESSED).insert('a', 1).insert('a', 2).insert('a', 3).insert('ab', 4).insert('ab', 5).insert('abc', 6)
        self.assertEqual([1, 2, 3, 4, 5, 6], sorted(t.get_all_values()))
        t.delete_prefix('ab')
        self.assertEqual([1, 2, 3], sorted(t.get_all_values()))
        t.delete_prefix('a')
        self.assertEqual(True, t.is_empty())
        return

    def test_clear(self):
        """Tests clear method"""
        t = Trie(MAKE_COMPRESSED).insert('a', 1).insert('ab', 2)
        self.assertEqual(2, t.size())
        t.clear()
        self.assertEqual(True, t.is_empty())
        return

    # TODO: add support for each_key, etc.
    #
    # # Test each_key.
    # def test_each_key(self):
    #   """Tests each_key"""
    #   t = Trie().insert('a', 1).insert('a', 2).insert('b', 3).insert('ab', 4)
    #   keys = []
    #   t.each_key {|k| keys.push(k.join) }
    #   self.assertEqual(['a', 'ab', 'b'], sorted(keys))
    #
    # # Test each_value.
    # def test_each_value(self):
    #   """"Tests each_value"""
    #   t = Trie().insert('a', 1).insert('a', 2).insert('b', 1)
    #   values = []
    #   t.each_value {|v| values.push(v) }
    #   self.assertEqual([1, 1, 2], sorted(values))
    #
    # # Test each.
    # def test_each(self):
    #   """Tests each"""
    #   t = Trie().insert('a', 1).insert('a', 2).insert('b', 3).insert('ab', 4)
    #   pairs = []
    #   t.each {|k, v| pairs.push([k.join, v]) }
    #   self.assertEqual([['a', 1], ['a', 2], ['ab', 4], ['b', 3]], sorted(pairs))
    #
    # # Test keys.
    # def test_keys(self):
    #   """Tests keys"""
    #   t = Trie().insert('a', 1).insert('a', 2).insert('abc', 3).insert('b', 4)
    #   keys = t.keys.collect {|k| k.join sorted(})
    #   self.assertEqual(['a', 'abc', 'b'], keys)

    def test_composition(self):
        """Test the composition of the tries by using the num_nodes method."""
        t = Trie(True).insert('a', 1)
        self.assertEqual(1, t.num_nodes())  # a
        t.insert('a', 2)
        self.assertEqual(1, t.num_nodes())  # a
        t.insert('abc', 3)
        self.assertEqual(3, t.num_nodes())  # '' -> a -> bc
        t.insert('ab', 4)
        self.assertEqual(4, t.num_nodes())  # '' -> a -> b -> c
        t.insert('b', 5)
        self.assertEqual(5, t.num_nodes())  # '' -> (a -> b -> c | b)
        t.insert('b', 6)
        self.assertEqual(5, t.num_nodes())  # '' -> (a -> b -> c | b)
        t.insert('abcdef', 7)
        self.assertEqual(6, t.num_nodes())  # '' -> (a -> b -> c -> def | b)
        t.insert('abcdeg', 8)
        # '' -> (a -> b -> c -> d -> e -> (f | g) | b)
        self.assertEqual(9, t.num_nodes())
        return

    # TODO: test serialization and string conversion
    #
    # def test_marshalling(self):
    #   """Tests Marshal.dump"""
    #   t = Trie().insert('a', 1).insert('a', 13).insert('at', 2).insert('b', 3)
    #   t2 = Marshal.load(Marshal.dump(t))
    #   self.assertEqual(t.size(), t2.size())
    #   %w(a at b x).each do |k|
    #     self.assertEqual(t[k], t2[k])
    #
    # def test_to_a(self):
    #   """Test to_a"""
    #   t = Trie().insert('a', 1).insert('a', 13).insert('at', 2).insert('b', 3)
    #   self.assertEqual([['a', 1], ['a', 13], ['at', 2], ['b', 3]],
    #                    t.to_a.map{|k,v| [k.join(''), v]})

#------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
