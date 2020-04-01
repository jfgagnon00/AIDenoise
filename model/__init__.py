import numpy as np
import pandas as pd
import torch.nn as torch

class FilterModel(torch.Module):
    def __init__(self, dimensionIn, dimensionOut):
        super().__init__()
        self._hidden1 = torch.Linear(dimensionIn, dimensionOut)
        self._hidden2 = torch.Linear(dimensionOut, 1)

    def forward(self, x):
        h1 = torch.functional.relu( self._hidden1(x) )
        out = torch.functional.relu( self._hidden2(h1) )
        return out

    def parameterCount(self):
        return np.sum(p.numel() for p in self.parameters() if p.requires_grad)
