import numpy as np
import pandas as pd
import torch

class Dataset(torch.utils.data.Dataset):
    """
    Wrap data fitlered with an custom moving average filter.
    Goal is to train a neural network that emulates this filter.
    """
    def __init__(self, csvFilename):
        super().__init__()
        data = pd.read_csv(csvFilename)
        self.cartIDs = data.CartID.to_numpy().astype(np.int32)
        self.rawPositions = data.RawPos.to_numpy().astype(np.float32)
        self.filteredPositions = data.MovingAverage.to_numpy().astype(np.float32)

    def __len__(self):
        return len(self.cartIDs)

    def __getitem__(self, index):
        x = self.rawPositions[index:index + 1]
        target = self.filteredPositions[index:index + 1]
        return x, target


if __name__ == "__main__":
    # tests to ensure data is properly loaded
    dataset = Dataset("./filtered_positions.csv")
    value = dataset.cartIDs[1496]
    assert value == 38

    value = dataset.rawPositions[1496]
    assert np.allclose(value, 0.434298)

    value = dataset.filteredPositions[1496]
    assert np.allclose(value, 0.429797275)
