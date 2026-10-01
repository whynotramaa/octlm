import unittest

from shop.text import count_words


class CountWordsTest(unittest.TestCase):
    def test_count_words(self):
        self.assertEqual(count_words("blue desk  lamp"), 3)
        self.assertEqual(count_words(""), 0)
