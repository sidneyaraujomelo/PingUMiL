class SetEdge:
    def __init__(self, origin, id_source, id_target, label_source, label_target):
        self.origin = origin
        self.id_source = id_source
        self.id_target = id_target
        self.label_source = label_source
        self.label_target = label_target

    def __str__(self):
        return "{}->{} ({}->{})".format(self.id_source, self.id_target, self.label_source, self.label_target)

    def compareType(self, other):
        return self.label_source == other.label_source and self.label_target == other.label_target

    def isType(self, label_source, label_target):
        return self.label_source == label_source and self.label_target == label_target