import unittest

from library.catalog import shelf_names


class ShelfNamesTest(unittest.TestCase):
    def test_shelf_names(self):
        self.assertEqual(shelf_names(), ["fiction", "history", "poetry", "science"])
