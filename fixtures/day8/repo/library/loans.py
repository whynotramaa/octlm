MAX_DAYS = 21
FINE_PER_DAY = 0.25


def due_in(days_out):
    return MAX_DAYS - days_out


def fine(days_late):
    return round(max(0, days_late) * FINE_PER_DAY, 2)


def can_borrow(count, limit=5):
    return count < limit
