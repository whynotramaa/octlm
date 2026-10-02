import unittest

from library.loans import overdue


class OverdueTest(unittest.TestCase):
    def test_overdue(self):
        self.assertFalse(overdue(21))
        self.assertTrue(overdue(22))
        self.assertFalse(overdue(0))
