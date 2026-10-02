import unittest

from library.text import snake_case


class SnakeCaseTest(unittest.TestCase):
    def test_snake_case(self):
        self.assertEqual(snake_case("Life of Pi"), "life_of_pi")
        self.assertEqual(snake_case("Middlemarch"), "middlemarch")
