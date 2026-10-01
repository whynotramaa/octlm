import unittest

from shop.inventory import total_stock


class TotalStockTest(unittest.TestCase):
    def test_total_stock(self):
        self.assertEqual(total_stock(), 46)
