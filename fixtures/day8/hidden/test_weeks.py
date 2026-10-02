import unittest

from library.dates import weeks


class WeeksTest(unittest.TestCase):
    def test_weeks(self):
        self.assertEqual(weeks(21), 3)
        self.assertEqual(weeks(20), 2)
        self.assertEqual(weeks(6), 0)
