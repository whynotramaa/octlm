import unittest

from shop.pricing import add_tax, apply_discount, round_money


class PricingTest(unittest.TestCase):
    def test_round_money(self):
        self.assertEqual(round_money(2.345678), 2.35)

    def test_apply_discount(self):
        self.assertEqual(apply_discount(80.0, 25), 60.0)

    def test_add_tax(self):
        self.assertEqual(add_tax(100.0), 108.0)
