import unittest

from shop.cart import Cart


class CartHasTest(unittest.TestCase):
    def test_has(self):
        cart = Cart()
        cart.add("mug", 4.5)
        self.assertTrue(cart.has("mug"))
        self.assertFalse(cart.has("lamp"))
