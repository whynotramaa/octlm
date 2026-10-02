import unittest

from library.members import Roster


class LookupTest(unittest.TestCase):
    def test_lookup(self):
        roster = Roster()
        roster.join(3, "Lena")
        self.assertEqual(roster.lookup(3), "Lena")
        self.assertIsNone(roster.lookup(9))
