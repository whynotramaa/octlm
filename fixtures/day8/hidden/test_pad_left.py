import unittest

from library.text import pad_left


class PadLeftTest(unittest.TestCase):
    def test_pad_left(self):
        self.assertEqual(pad_left("ab", 4), "  ab")
        self.assertEqual(pad_left("abcdef", 3), "abcdef")
