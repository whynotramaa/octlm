from shop.pricing import round_money


class Cart:
    def __init__(self):
        self.items = {}

    def add(self, name, price, quantity=1):
        if quantity < 1:
            raise ValueError("quantity must be positive")
        _, held = self.items.get(name, (price, 0))
        self.items[name] = (price, held + quantity)

    def remove(self, name):
        self.items.pop(name, None)

    def item_count(self):
        return sum(quantity for _, quantity in self.items.values())

    def total(self):
        return round_money(sum(price * quantity for price, quantity in self.items.values()))
