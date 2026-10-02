import unittest

from library.loans import late_days


class LateDaysTest(unittest.TestCase):
    def test_late_days(self):
        self.assertEqual(late_days(25), 4)
        self.assertEqual(late_days(10), 0)
