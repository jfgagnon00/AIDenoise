import itertools
import numpy as np
import os
import pandas as pd
import torch

class Dataset(torch.utils.data.Dataset):
    """
    Wrap data fitlered with an custom moving average filter.
    Goal is to train a neural network that emulates this filter.
    """
    def __init__(self, csvFilenames):
        # remap data to [0, 1]
        # acquisition process has shown original data should be in [-0.2, 1.2]
        min = -0.2
        max = 1.2
        self.scale = 1.0 / (max - min)
        self.bias = -min * self.scale

        self.features = None
        self.targets = None

        for csvFilename in csvFilenames:
            features, targets = self._loadCsv(csvFilename)
            if self.features is None:
                self.features = features
                self.targets = targets
            else:
                self.features = np.append(self.features, features)
                self.targets = np.append(self.targets, targets)

        # remap features[?, 1]/labels to [0, 1]
        self.features[:, 1] = self.features[:, 1] * self.scale + self.bias
        self.targets = self.targets * self.scale + self.bias

    def numFeatures(self):
        return len(self.features.shape)

    def preprocess(self, sequenceLength, device):
        # wrap features and labels for device (possibly making a copy)
        self._featuresDevice = torch.from_numpy(self.features).to(device)
        self._targetsDevice = torch.from_numpy(self.targets).to(device)

        # setup values and labels
        self._data = []
        end = len(self._featuresDevice) - sequenceLength - 1
        for i in range(end):
            start = i
            stop = i + sequenceLength

            feature = self._featuresDevice[start:stop,:]
            label = self._targetsDevice[stop:stop + 1]

            self._data.append((feature, label))

    def _loadCsv(self, csvFilename):
        # fetch data from csvFilename
        data = pd.read_csv(csvFilename)

        # extract features from data
        # shape is (num_samples, 2)
        features = data[["CartID", "RawPos"]].to_numpy().astype(np.float32)

        # extract target from data
        # shape is (num_samples,)
        targets = data.MovingAverage.to_numpy().astype(np.float32)

        return features, targets

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


def isolateCarts(csvFilenameIn):
    """
    Split time series into cart specific segments
    """
    data = pd.read_csv(csvFilenameIn)

    def hasCartChanged(rowIndex):
        if (rowIndex > 0):
            return data.at[rowIndex, "RawPos"] < -0.1 or data.at[rowIndex, "CartID"] != data.at[rowIndex - 1, "CartID"]
        else:
            return True

    # find indices where cart changed
    dataCount = data.shape[0]
    cartChangedIndices = [x for x in range(dataCount) if hasCartChanged(x)]

    # filter time series "segments" to keep only real cart changes
    indicesCount = len(cartChangedIndices)
    validSegments = []
    for x in range(1, indicesCount):
        segmentStart = cartChangedIndices[x - 1]
        segmentStop = cartChangedIndices[x]
        if data.at[segmentStop, "CartID"] != data.at[segmentStart, "CartID"]:
            validSegments.append( (segmentStart, segmentStop) )

    # edge case: no cart change on last segment
    segmentStart = cartChangedIndices[-1]
    segmentStop = dataCount - 1
    if data.at[segmentStop, "CartID"] == data.at[segmentStart, "CartID"] and (segmentStop - segmentStart > 10):
        validSegments.append( (segmentStart, segmentStop) )

    # save 1 csv file per cart
    name, ext = os.path.splitext(csvFilenameIn)
    for segmentIndex, (segmentStart, segmentStop) in enumerate(validSegments):
        segment = data.iloc[segmentStart: segmentStop]
        outFilename = f"{name}_{segmentIndex:02d}{ext}"
        segment.to_csv(outFilename, encoding='utf-8', index=False)


if __name__ == "__main__":
    isolateCarts("./data/filtered_positions.csv")
