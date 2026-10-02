from library.text import format_title

SHELVES = {"fiction": 120, "history": 45, "science": 60, "poetry": 8}


def shelf_size(name):
    return SHELVES.get(name, 0)


def small_shelves(limit=10):
    return sorted(name for name, count in SHELVES.items() if count < limit)


def add_books(name, count):
    SHELVES[name] = SHELVES.get(name, 0) + count
    return SHELVES[name]


def label(title, shelf):
    return f"{format_title(title)} [{shelf}]"
