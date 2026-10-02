def format_title(title):
    return " ".join(word.capitalize() for word in title.split())


def initials(name):
    return "".join(part[0].upper() for part in name.split())


def shorten(text, width=20):
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."
