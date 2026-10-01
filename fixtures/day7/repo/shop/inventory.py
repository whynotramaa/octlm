STOCK = {"lamp": 12, "chair": 4, "table": 0, "mug": 30}


def in_stock(name):
    return STOCK.get(name, 0) > 0


def restock(name, count):
    STOCK[name] = STOCK.get(name, 0) + count
    return STOCK[name]


def low_stock(threshold=5):
    return sorted(name for name, count in STOCK.items() if count < threshold)
