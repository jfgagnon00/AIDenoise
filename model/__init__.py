import itertools
import torch
import numpy as np


class LSTMFilter(torch.nn.Module):
    """
    Model a time sequence to estimate a custom filtering
    algorithm.
    """
    def __init__(self, device, dimensionIn=1, dimensionHidden=64, dimensionOut=1, numLayers=1, batchSize=1):
        super().__init__()

        self._device = device
        self._numLayers = 1

        # self._dimensionIn = [dimensionIn, dimensionHidden, dimensionHidden // 2]
        # self._dimensionHidden = [dimensionHidden, dimensionHidden // 2, dimensionHidden]
        self._dimensionIn = [dimensionIn]
        self._dimensionHidden = [dimensionHidden]

        self._lstms = []
        for dimensionIn, dimensionHidden in itertools.zip_longest(self._dimensionIn, self._dimensionHidden):
            layer = torch.nn.LSTM(dimensionIn, dimensionHidden, batch_first=True).to(device)
            self._lstms.append(layer)
        self._net = torch.nn.Sequential(*self._lstms)

    def getName(self):
        return "LSTMFilter"

    def resetSequence(self, batchSize):
        self._batchSize = batchSize
        self._hiddens = []
        for dimensionHidden in self._dimensionHidden:
            hidden = (torch.zeros(self._numLayers, self._batchSize, dimensionHidden).to(self._device),
                      torch.zeros(self._numLayers, self._batchSize, dimensionHidden).to(self._device))
            self._hiddens.append(hidden)

    def forward(self, x):
        # sanity checks on x
        # expects shape=(batch, sequence_len, dimensionIn)
        assert x.dim() == 2 or x.dim() == 3

        if x.dim() == 2:
            # remap x to fit requirement of LSTM module
            seqLen, dimIn = x.shape
            x = x.view(1, seqLen, dimIn)

        # do some more sanity checks
        assert x.shape[0] == self._batchSize
        assert x.shape[2] == self._dimensionIn[0]

        # evaluate LSTM modules
        out = x
        for lstm, hidden in itertools.zip_longest(self._lstms, self._hiddens):
            out, hidden = lstm(out, hidden)

            if lstm == self._lstms[-1]:
                # out.shape = (batch, sequence_len, dimensionHidden)
                # only keep last element of sequence dimension since this
                # is our desired prediction
                out = out[:, :, -1]
            else:
                # apply activation
                out = torch.nn.functional.leaky_relu(out)

        return out


class ConvAutoEncoderFilter(torch.nn.Module):
    """
    Model an auto encoder using convolution to estimate a custom filtering
    algorithm.
    """
    def __init__(self, device, sequenceLength, numFeatures, encoderDimensions, kernelSizes):
        super().__init__()

        layers, outFeatures = self._getLayers(sequenceLength, numFeatures, encoderDimensions, kernelSizes)

        # adapt output of autoencoder to desired output size
        # no dropout/activation here
        layer = torch.nn.Linear(outFeatures, sequenceLength)
        layers.append(layer)

        self._net = torch.nn.Sequential(*layers).to(device)

    def _getLayers(self, sequenceLength, numChannels, encoderDimensions, kernelSizes):
        layers = []
        encoderDimensions = encoderDimensions + encoderDimensions[1::-1]
        kernelSizes = kernelSizes + kernelSizes[1::-1]

        inFeatures = numChannels
        for outFeatures, kernelSize in itertools.zip_longest(encoderDimensions, kernelSizes):
            sequenceLength -= kernelSize - 1

            layer = torch.nn.Conv1d(inFeatures, outFeatures, kernelSize)
            layers.append(layer)

            activation = torch.nn.LeakyReLU()
            layers.append(activation)

            inFeatures = outFeatures

        # adapt output of conv autoencoder to be a 2d shape
        layer = torch.nn.Conv1d(inFeatures, out_channels=1, kernel_size=1)
        layers.append(layer)

        activation = torch.nn.LeakyReLU()
        layers.append(activation)

        flatten = torch.nn.Flatten()
        layers.append(flatten)

        return layers, sequenceLength

    def getName(self):
        return "ConvAutoEncoderFilter"

    def resetSequence(self, batchSize):
        pass

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self._net(x)


class DenseFilter(torch.nn.Module):
    """
    Estimate a custom filtering algorithm using Linear
    layers
    """
    def __init__(self, device, sequenceLength, numFeatures, dimensions, outputMatchSequenceLength=False):
        super().__init__()

        layers, outFeatures = self._getLayers(sequenceLength, numFeatures, dimensions)

        if outputMatchSequenceLength and outFeatures != sequenceLength:
            # adapt output to desired match sequenceLength
            # no dropout/activation here
            layer = torch.nn.Linear(outFeatures, sequenceLength)
            layers.append(layer)

        self._net = torch.nn.Sequential(*layers).to(device)

    def _getLayers(self, sequenceLength, numFeatures, dimensions):
        layers = []

        # adapt input (dimensionIn, numFeatures) to be (dimensionIn, 1)
        layer = torch.nn.Flatten()
        layers.append(layer)

        # apply autoencoder
        dimensions = [sequenceLength * numFeatures] + dimensions
        size = len(dimensions) - 1
        for i in range(size):
            inFeatures = dimensions[i]
            outFeatures = dimensions[i + 1]

            layer = torch.nn.Linear(inFeatures, outFeatures)
            layers.append(layer)

            if i < size - 1:
                activation = torch.nn.LeakyReLU()
                # activation = torch.nn.ReLU()  # worst ever
                # activation = torch.nn.Sigmoid()
                # activation = torch.nn.Tanh()
                layers.append(activation)
                pass

        return layers, outFeatures

    def getName(self):
        return "DenseFilter"

    def resetSequence(self, batchSize=None):
        pass

    def forward(self, x):
        return self._net(x)


class SimpleLinear(torch.nn.Module):
    """
    Estimate a custom filtering algorithm using 1 simple Linear layer
    """
    def __init__(self, device, sequenceLength, numFeatures):
        super().__init__()
        layers = self._getLayers(sequenceLength, numFeatures)
        self._net = torch.nn.Sequential(*layers).to(device)

    def _getLayers(self, sequenceLength, numFeatures):
        layers = []

        # numLayer = [sequenceLength, 100, 33]
        numLayer = [sequenceLength, 20]
        outFeature = sequenceLength

        for inFeature, outFeature in zip(numLayer, numLayer[1:]):
            layer = torch.nn.Linear(inFeature, outFeature, bias=False)
            # torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.kaiming_uniform_(layer.weight)
            layers.append(layer)

            # activation = torch.nn.PReLU()
            # layers.append(activation)

            # drop = torch.nn.Dropout(p=0.33)
            # layers.append(drop)

            # norm = torch.nn.BatchNorm1d(outFeature)
            # layers.append(norm)

        layer = torch.nn.Linear(outFeature, 1, bias=False)
        layers.append(layer)

        return layers

    def getName(self):
        return "SimpleLinear"

    def resetSequence(self, batchSize=None):
        pass

    def forward(self, x, pos):
        return self._net(x)

    def debugPrint(self):
        for p in self._net.parameters():
            print(f"{p.data}")


class BasicRNN(torch.nn.Module):
    """
    Estimate a custom filtering algorithm using 1 hidden state
    """
    def __init__(self, device, sequenceLength, hiddenLength, numFeatures):
        super().__init__()
        layers = self._getLayers(sequenceLength, hiddenLength, numFeatures)
        self._h = None
        self._device = device
        self._net = torch.nn.Sequential(*layers).to(device)

    def _getLayers(self, sequenceLength, hiddenLength, numFeatures):
        outFeature = 1

        layers = []

        layer = torch.nn.Linear(sequenceLength + hiddenLength, 1, bias=True)
        torch.nn.init.xavier_uniform_(layer.weight)
        layers.append(layer)

        # layer = torch.nn.Linear(2, 1, bias=False)
        # torch.nn.init.xavier_uniform_(layer.weight)
        # layers.append(layer)

        # torch.nn.init.kaiming_uniform_(layer.weight)

        return layers

    def getName(self):
        return "BasicRNN"

    def resetSequence(self, batchSize=None):
        self._h = np.zeros((batchSize, 1), dtype=np.float32)
        self._h = torch.from_numpy(self._h).to(self._device)

    def forward(self, x, pos):
        tmpRows = []
        for batch, hidden in zip(x, self._h):
            tmp = torch.cat([batch, hidden])
            tmpRows.append(tmp)
        x = torch.stack(tmpRows)
        y = self._net(x)
        self._h = y
        return y

    def debugPrint(self):
        pass
