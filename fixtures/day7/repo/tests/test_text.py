import unittest

from shop.text import slugify, title_case, truncate


class TextTest(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_title_case(self):
        self.assertEqual(title_case("blue desk lamp"), "Blue Desk Lamp")

    def test_truncate(self):
        self.assertEqual(truncate("abcdefghij", 8), "abcde...")
        self.assertEqual(truncate("short", 8), "short")
