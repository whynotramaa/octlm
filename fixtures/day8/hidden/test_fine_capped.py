import unittest

from library.loans import fine_capped


class FineCappedTest(unittest.TestCase):
    def test_fine_capped(self):
        self.assertEqual(fine_capped(4), 1.0)
        self.assertEqual(fine_capped(40), 5.0)
        self.assertEqual(fine_capped(40, cap=2.0), 2.0)
