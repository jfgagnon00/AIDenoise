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
    NumFeatures = 1

    def __init__(self, csvFilenames):
        self._features = None
        self._labels = None
        for csvFilename in csvFilenames:
            features, labels = self._loadCsv(csvFilename)
            if self._features is None:
                self._features = features
                self._labels = labels
            else:
                self._features = np.append(self._features, features)
                self._labels = np.append(self._labels, labels)

    def preprocess(self, device, sequenceLength):
        self._featuresDevice = torch.from_numpy(self._features).to(device)
        self._labelsDevice = torch.from_numpy(self._labels).to(device)

        size = len(self._features)
        featuresViews = []
        labelViews = []
        for i in range(size - sequenceLength - 1):
            start = i
            stop = i + sequenceLength

            featuresView = self._featuresDevice[start:stop]
            labelView = self._labelsDevice[stop:stop + 1]

            featuresViews.append(featuresView)
            labelViews.append(labelView)

        self._features = featuresViews
        self._labels = labelViews
        self.sequenceLengthIn = sequenceLengthIn
        self.sequenceLengthOut = sequenceLengthOut

    def applyScaleBias(self, scale, bias, useData=False):
        if useData:
            # remap data of interrest in [0, 1]
            # is it needed, or in [-1, 1]
            min = np.amin(self._features)
            max = np.amax(self._features)

            self.scale = 1.0 / (max - min)
            self.bias = -min * self.scale
        else:
            self.scale = scale
            self.bias = bias

        if Dataset.NumFeatures == 1:
            self._features = self._features * self.scale + self.bias
        else:
            self._features[:, 1] = self._features[:, 1] * self.scale + self.bias

        self._labels = self._labels * self.scale + self.bias

    def numFeatures(self):
        return len(self.features.shape)

    def preprocess(self, device, sequenceLengthIn, sequenceLengthOut):
        # wrap features and labels for device (possibly making a copy)
        self._featuresDevice = torch.from_numpy(self._features).to(device)
        self._labelsDevice = torch.from_numpy(self._labels).to(device)

        # setup values and labels
        self._data = []
        end = len(self._featuresDevice) - sequenceLengthIn - 1
        for i in range(end):
            start = i
            stop = i + sequenceLengthIn

            if Dataset.NumFeatures == 1:
                features = self._featuresDevice[start:stop]
            else:
                features = self._featuresDevice[start:stop,:]
            label = self._labelsDevice[stop:stop + 1]

            self._data.append((features, label))

    def _loadCsv(self, csvFilename):
        # fetch data from csvFilename
        data = pd.read_csv(csvFilename)

        # extract features from data
        if Dataset.NumFeatures == 1:
            # shape is (num_samples,)
            features = data["RawPos"].to_numpy().astype(np.float32)
        else:
            # shape is (num_samples, 2)
            features = data[["CartID", "RawPos"]].to_numpy().astype(np.float32)

        # extract label from data
        # shape is (num_samples,)
        labels = data.MovingAverage.to_numpy().astype(np.float32)

        return features, labels

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


def isolateCarts(csvFilenameIn):
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
    dataset = Dataset("./data/filtered_positions.csv")
