import unittest

from library.catalog import largest_shelf


class LargestShelfTest(unittest.TestCase):
    def test_largest_shelf(self):
        self.assertEqual(largest_shelf(), "fiction")
