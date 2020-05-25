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

    def __init__(self, csvFilename, sequenceLength=1):
        self._load(csvFilename)

    def preprocess(self, device, sequenceLengthIn, sequenceLengthOut):
        self._featuresDevice = torch.from_numpy(self._features).to(device)
        self._targetsDevice = torch.from_numpy(self._targets).to(device)

        size = len(self._features)
        featureViews = []
        targetViews = []
        for i in range(size - sequenceLengthIn - 1):
            start = i
            stop = i + sequenceLengthIn

            featureView = self._featuresDevice[start:stop]
            # targetView = self._targetsDevice[start + 1:stop + 1] if useLSTM else self._targetsDevice[start:stop]

            # start = stop - sequenceLengthOut
            # targetView = self._targetsDevice[start:stop]
            targetView = self._targetsDevice[stop:stop + 1]

            featureViews.append(featureView)
            targetViews.append(targetView)

        self._features = featureViews
        self._targets = targetViews
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

        self._features = self._features * self.scale + self.bias
        self._targets = self._targets * self.scale + self.bias

    def _load(self, csvFilename):
        # load data of interest
        # remap data to [0, 1]
        self._sequenceLength = sequenceLength
        min = np.amin(cartIDs)
        max = np.amax(cartIDs)
        cartIDs = cartIDs - min
        cartIDs = cartIDs / (max - min)
        min = -0.2
        self.cartIDs = data.CartID.to_numpy().astype(np.int32)
        self.rawPositions = data.RawPos.to_numpy().astype(np.float32)
        max = 1.2
        # pack data to map
        min = np.amin(self.rawPositions)
        max = np.amax(self.rawPositions)
        for i, (cartID, rawPosition) in enumerate(itertools.zip_longest(cartIDs, rawPositions)):
            self._features[i] = rawPosition
            # self._features[i, 0] = rawPosition
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
        self.rawPositions = self.rawPositions * self.scale + self.bias
        self.filteredPositions = self.filteredPositions * self.scale + self.bias

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
        return len(self.cartIDs) - self._sequenceLength - 1

    def __getitem__(self, index):
        start = index

def isolateCarts(csvFilenameIn):
    """
        x = self.rawPositions[start:stop]
    """
    data = pd.read_csv(csvFilenameIn)
        target = self.filteredPositions[stop:stop + 1]
    def hasCartChanged(rowIndex):
        if (rowIndex > 0):
            return data.at[rowIndex, "RawPos"] < -0.1 or data.at[rowIndex, "CartID"] != data.at[rowIndex - 1, "CartID"]
        else:
        return x, target
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


    dataset = Dataset("./filtered_positions.csv")
