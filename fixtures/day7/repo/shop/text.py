import re


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def title_case(text):
    return " ".join(word.capitalize() for word in text.split())


def truncate(text, limit):
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
