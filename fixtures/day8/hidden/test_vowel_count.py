import unittest

from library.text import vowel_count


class VowelCountTest(unittest.TestCase):
    def test_vowel_count(self):
        self.assertEqual(vowel_count("Odyssey"), 2)
        self.assertEqual(vowel_count("AEIOU xyz"), 5)
        self.assertEqual(vowel_count(""), 0)
