import unittest

from shop.pricing import percent_off


class PercentOffTest(unittest.TestCase):
    def test_percent_off(self):
        self.assertEqual(percent_off(100.0, 75.0), 25.0)
        self.assertEqual(percent_off(80.0, 60.0), 25.0)
