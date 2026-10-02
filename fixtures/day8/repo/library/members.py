class Roster:
    def __init__(self):
        self.members = {}

    def join(self, member_id, name):
        if member_id < 1:
            raise ValueError("member id must be positive")
        self.members[member_id] = name

    def leave(self, member_id):
        self.members.pop(member_id, None)

    def size(self):
        return len(self.members)

    def names(self):
        return sorted(self.members.values())
