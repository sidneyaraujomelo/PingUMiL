import torch.utils.data.Dataset

class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, z, edges, labels):
        'initialization'
        self.labels = labels
        self.edges = edges
        self.z = z

    def __len__(self):
        'Denotes the total number of samples'
        return self.edges.size()[-1]

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        x = self.edges[:,index].reshape(-1,1)
        y = self.labels[index]
        return x, y 
