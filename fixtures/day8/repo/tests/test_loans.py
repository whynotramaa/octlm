import unittest

from library.loans import can_borrow, due_in, fine


class LoansTest(unittest.TestCase):
    def test_due_in(self):
        self.assertEqual(due_in(5), 16)

    def test_fine(self):
        self.assertEqual(fine(4), 1.0)
        self.assertEqual(fine(-2), 0)

    def test_can_borrow(self):
        self.assertTrue(can_borrow(4))
        self.assertFalse(can_borrow(5))
