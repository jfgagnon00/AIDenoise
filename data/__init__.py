import numpy as np
import pandas as pd
import torch

class Dataset(torch.utils.data.Dataset):
    _CARTID = 0
    _INDEX = 1
    _RAWPOS = 2
    _FITLEREDPOS = 3

    def __init__(self, csvFilename):
        super().__init__()
        data = pd.read_csv(csvFilename)
        self.cartIDs = data.CartID.to_numpy().astype(np.int)
        self.rawPositions = data.RawPos.to_numpy().astype(np.float32)
        self.filteredPositions = data.MovingAverage.to_numpy().astype(np.float32)

    def __len__(self):
        return len(self.cartIDs)

    def __getitem__(self, index):
        x = self.rawPositions[index:index + 1]
        target = self.filteredPositions[index:index + 1]
        return x, target


if __name__ == "__main__":
    dataset = Dataset("./filtered_positions.csv")
    value = dataset.cartIDs[1496]
    assert value == 38

    value = dataset.rawPositions[1496]
    assert value == 0.434298

    value = dataset.filteredPositions[1496]
    assert value == 0.429797275
