import numpy as np
import torch
import torch.nn.functional as F

class MultilabelClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultilabelClassifier, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)
    
    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, sigmoid=False):
        z = self.lin(x)
        return torch.sigmoid(z) if sigmoid else z