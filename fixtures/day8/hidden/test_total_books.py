import unittest

from library.catalog import total_books


class TotalBooksTest(unittest.TestCase):
    def test_total_books(self):
        self.assertEqual(total_books(), 233)
