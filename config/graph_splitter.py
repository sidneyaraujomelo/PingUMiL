class GraphSplitter():
    def __init__(self, train_ratio, train_dict, test_ratio, test_dict, val_ratio, val_dict):
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.train_dict = train_dict
        self.test_dict = test_dict
        self.val_dict = val_dict

    def get_set_element(self, a):
        return None

class NodeSplitter(GraphSplitter):
    def get_set_element(self, a):
        if a <= self.train_ratio:
            return self.train_dict
        elif a <= self.train_ratio + self.test_ratio + 0.001:
            return self.test_dict
        return self.val_dict

class NodeGatSplitter(GraphSplitter):
    def get_set_element(self, a):
        if a <= self.train_ratio:
            return self.test_dict
        elif a <= self.test_ratio + self.val_ratio:
            return self.val_dict
        return None

class NodeGatMultigraphSplitter(GraphSplitter):
    def get_set_element(self, a):
        if a <= self.train_ratio:
            return self.train_dict
        elif a <= self.train_ratio + self.test_ratio + 0.001:
            return self.test_dict
        return self.val_dict

class EdgeSplitter(GraphSplitter):
    def get_set_element(self, a):
        if a <= self.train_ratio:
            return self.train_dict
        elif a <= self.train_ratio + self.test_ratio + 0.001:
            return self.test_dict
        return self.val_dict
