import unittest

from shop.text import reverse_words


class ReverseWordsTest(unittest.TestCase):
    def test_reverse_words(self):
        self.assertEqual(reverse_words("blue desk lamp"), "lamp desk blue")
