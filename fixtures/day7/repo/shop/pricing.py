TAX_RATE = 0.08


def round_money(amount):
    return round(amount, 2)


def apply_discount(price, percent):
    return round_money(price * (1 - percent / 100))


def add_tax(price, rate=TAX_RATE):
    return round_money(price * (1 + rate))
