import numpy as np
import pandas as pd
import torch


class Dataset(torch.utils.data.Dataset):
    """
    Wrap data fitlered with an custom moving average filter.
    Goal is to train a neural network that emulates this filter.
    """
    def __init__(self, csvFilename, sequenceLength=1):
        super().__init__()
        data = pd.read_csv(csvFilename)

        self._sequenceLength = sequenceLength

        # todo: data should be normalized [-1, 1]
        #       at the moment, [-0.2, 1.2]
        self.cartIDs = data.CartID.to_numpy().astype(np.int32)
        self.rawPositions = data.RawPos.to_numpy().astype(np.float32)
        self.filteredPositions = data.MovingAverage.to_numpy().astype(np.float32)

        min = np.amin(self.rawPositions)
        max = np.amax(self.rawPositions)
        self.scale = 2.0 / (max - min)
        self.bias = -min * self.scale - 1.0

        self.rawPositions = self.rawPositions * self.scale + self.bias
        self.filteredPositions = self.filteredPositions * self.scale + self.bias

    def __len__(self):
        return len(self.cartIDs) - self._sequenceLength - 1

    def __getitem__(self, index):
        start = index
        stop = index + self._sequenceLength

        x = self.rawPositions[start:stop]
        target = self.filteredPositions[stop:stop + 1]

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
