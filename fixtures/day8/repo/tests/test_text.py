import unittest

from library.text import format_title, initials, shorten


class TextTest(unittest.TestCase):
    def test_format_title(self):
        self.assertEqual(format_title("life of  pi"), "Life Of Pi")

    def test_initials(self):
        self.assertEqual(initials("jane austen"), "JA")

    def test_shorten(self):
        self.assertEqual(shorten("short"), "short")
        self.assertEqual(shorten("a" * 30, 10), "aaaaaaa...")
        self.assertEqual(len(shorten("b" * 25)), 20)
