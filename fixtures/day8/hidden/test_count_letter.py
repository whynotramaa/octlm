import unittest

from library.text import count_letter


class CountLetterTest(unittest.TestCase):
    def test_count_letter(self):
        self.assertEqual(count_letter("Anna Karenina", "a"), 4)
        self.assertEqual(count_letter("Odyssey", "Y"), 2)
        self.assertEqual(count_letter("Pi", "z"), 0)
