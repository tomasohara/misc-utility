#! /usr/bin/env python
#
# Test(s) for tpo_wordnet.py
#
# TODO:
# - get_word_spec(word, part_of_speech=None)
# - parse_wordform(wordform)
# - get_part_of_speech(tag, default=UNKNOWN_PART_OF_SPEECH)
# - get_synset(synset_spec)
# - get_lemma_word(lemma, prefix="")
# - get_synset_words(synset, prefix="")
#

import unittest
import tpo_wordnet as wn

class TestIt(unittest.TestCase):
    """Class for testcase definition"""

    def test_get_part_of_speech_prefix(self):
        """Test for get_part_of_speech_prefix(part_of_speech=None)"""
        self.assertEqual("", wn.get_part_of_speech_prefix())
        self.assertEqual("n:", wn.get_part_of_speech_prefix("n"))
        self.assertEqual("adv:", wn.get_part_of_speech_prefix("adv"))
        return

    def test_get_root_word(self):
        """Tests for get_root_word(wordform, [part_of_speech=None])"""
        self.assertEqual(wn.get_root_word("sings"), "sing")
        self.assertEqual(wn.get_root_word("pants", "n"), "pants")
        self.assertEqual(wn.get_root_word("gambling"), "gambling")
        return

    def test_get_synonyms(self):
        """Test for get_synonyms(wordform)"""
        self.assertTrue(set(["cad", "hound", "frump", "hotdog"]).issubset(set(wn.get_synonyms("dog"))))
        self.assertFalse("v:hound" in wn.get_synonyms("v:dog"))
        return

    def test_get_synset_hypernyms(self):
        """Test for get_synset_hypernyms(synset, [max_dist=1, processed=None, indent=""])"""
        self.assertTrue(wn.get_synset("hound.n.01") in wn.get_synset_hypernyms(wn.get_synset("beagle.n.01")))
        self.assertTrue(wn.get_synset("canine.n.02") in wn.get_synset_hypernyms(wn.get_synset("hound.n.01"), max_dist=3))
        return

    def test_get_hypernym_terms(self):
        """Test for get_hypernym_terms(wordform, [max_dist=1])"""
        self.assertTrue("n:hound" in wn.get_hypernym_terms("n:beagle"))
        self.assertFalse("n:mammal" in wn.get_hypernym_terms("n:dog"))
        self.assertTrue("n:mammal" in wn.get_hypernym_terms("n:dog", 
                                                             max_dist=5))
        return

#------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
