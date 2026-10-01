import unittest

from shop.sku import is_valid_sku


class SkuTest(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(is_valid_sku("ABC-1234"))

    def test_invalid(self):
        for sku in ("abc-1234", "AB-1234", "ABC-123", "ABC1234", "ABC-12345"):
            self.assertFalse(is_valid_sku(sku), sku)
