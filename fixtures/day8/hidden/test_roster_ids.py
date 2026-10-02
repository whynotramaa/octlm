import unittest

from library.members import Roster


class RosterIdsTest(unittest.TestCase):
    def test_ids(self):
        roster = Roster()
        roster.join(3, "Lena")
        roster.join(1, "Ines")
        self.assertEqual(roster.ids(), [1, 3])
