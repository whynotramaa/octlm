import unittest

from library import catalog


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.saved = dict(catalog.SHELVES)

    def tearDown(self):
        catalog.SHELVES.clear()
        catalog.SHELVES.update(self.saved)

    def test_shelf_size(self):
        self.assertEqual(catalog.shelf_size("history"), 45)
        self.assertEqual(catalog.shelf_size("maps"), 0)

    def test_small_shelves(self):
        self.assertEqual(catalog.small_shelves(), ["poetry"])

    def test_add_books(self):
        self.assertEqual(catalog.add_books("poetry", 2), 10)

    def test_label(self):
        self.assertEqual(catalog.label("the odyssey", "fiction"), "The Odyssey [fiction]")
