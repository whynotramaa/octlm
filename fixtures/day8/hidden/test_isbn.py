import unittest

from library.isbn import is_isbn13


class IsbnTest(unittest.TestCase):
    def test_is_isbn13(self):
        self.assertTrue(is_isbn13("9780140449136"))
        self.assertFalse(is_isbn13("978014044913"))
        self.assertFalse(is_isbn13("97801404491X6"))
        self.assertFalse(is_isbn13(""))
