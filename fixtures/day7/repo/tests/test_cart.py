import unittest

from shop.cart import Cart


class CartTest(unittest.TestCase):
    def test_total_counts_quantity(self):
        cart = Cart()
        cart.add("mug", 4.5, 2)
        cart.add("lamp", 20.0)
        self.assertEqual(cart.total(), 29.0)

    def test_item_count(self):
        cart = Cart()
        cart.add("mug", 4.5, 2)
        cart.add("mug", 4.5)
        self.assertEqual(cart.item_count(), 3)

    def test_remove(self):
        cart = Cart()
        cart.add("mug", 4.5)
        cart.remove("mug")
        self.assertEqual(cart.total(), 0)
