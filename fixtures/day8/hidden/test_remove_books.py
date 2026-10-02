import unittest

from library import catalog


class RemoveBooksTest(unittest.TestCase):
    def setUp(self):
        self.saved = dict(catalog.SHELVES)

    def tearDown(self):
        catalog.SHELVES.clear()
        catalog.SHELVES.update(self.saved)

    def test_remove_books(self):
        self.assertEqual(catalog.remove_books("poetry", 3), 5)
        self.assertEqual(catalog.remove_books("poetry", 9), 0)
