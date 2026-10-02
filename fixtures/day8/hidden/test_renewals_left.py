import unittest

from library.loans import renewals_left


class RenewalsTest(unittest.TestCase):
    def test_renewals_left(self):
        self.assertEqual(renewals_left(0), 3)
        self.assertEqual(renewals_left(2), 1)
        self.assertEqual(renewals_left(5), 0)
