import unittest

from library.members import Roster


class RenameTest(unittest.TestCase):
    def test_rename(self):
        roster = Roster()
        roster.join(1, "Ines")
        roster.rename(1, "Inez")
        self.assertEqual(roster.names(), ["Inez"])
        with self.assertRaises(KeyError):
            roster.rename(7, "Nobody")
