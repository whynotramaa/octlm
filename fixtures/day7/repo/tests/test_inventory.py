import unittest

from shop import inventory


class InventoryTest(unittest.TestCase):
    def setUp(self):
        self.saved = dict(inventory.STOCK)

    def tearDown(self):
        inventory.STOCK.clear()
        inventory.STOCK.update(self.saved)

    def test_in_stock(self):
        self.assertTrue(inventory.in_stock("chair"))
        self.assertFalse(inventory.in_stock("table"))

    def test_restock(self):
        self.assertEqual(inventory.restock("table", 3), 3)

    def test_low_stock(self):
        self.assertEqual(inventory.low_stock(), ["chair", "table"])
