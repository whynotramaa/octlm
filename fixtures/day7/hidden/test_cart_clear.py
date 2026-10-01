import unittest

from shop.cart import Cart


class CartClearTest(unittest.TestCase):
    def test_clear(self):
        cart = Cart()
        cart.add("mug", 4.5, 2)
        cart.clear()
        self.assertEqual(cart.item_count(), 0)
        self.assertEqual(cart.total(), 0)
