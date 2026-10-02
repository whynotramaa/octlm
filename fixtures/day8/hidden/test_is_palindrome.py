import unittest

from library.text import is_palindrome


class PalindromeTest(unittest.TestCase):
    def test_is_palindrome(self):
        self.assertTrue(is_palindrome("Never odd or even"))
        self.assertTrue(is_palindrome("abba"))
        self.assertFalse(is_palindrome("library"))
