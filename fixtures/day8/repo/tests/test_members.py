import unittest

from library.members import Roster


class RosterTest(unittest.TestCase):
    def test_join_and_size(self):
        roster = Roster()
        roster.join(1, "Ines")
        roster.join(2, "Kofi")
        self.assertEqual(roster.size(), 2)

    def test_join_rejects_zero(self):
        with self.assertRaises(ValueError):
            Roster().join(0, "Nobody")

    def test_names_sorted(self):
        roster = Roster()
        roster.join(2, "Kofi")
        roster.join(1, "Ines")
        self.assertEqual(roster.names(), ["Ines", "Kofi"])

    def test_leave(self):
        roster = Roster()
        roster.join(1, "Ines")
        roster.leave(1)
        self.assertEqual(roster.size(), 0)
