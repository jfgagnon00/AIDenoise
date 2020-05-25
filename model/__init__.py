import numpy as np
import torch


class FilterModel(torch.nn.Module):
    """
    Model a time sequence to estimate a custom filtering
    algorithm.
    """
    def __init__(self, device, dimensionIn=1, dimensionHidden=64, dimensionOut=1):
        super().__init__()

        self._device = device
        self._dimensionIn = dimensionIn
        self._dimensionHidden = dimensionHidden
        self._numLayers = 1

        self._lstm = torch.nn.LSTM(dimensionIn, dimensionHidden, batch_first=True).to(device)
        self._hiddenToPrediction1 = torch.nn.Linear(dimensionHidden // 1, dimensionHidden // 2).to(device)
        self._hiddenToPrediction2 = torch.nn.Linear(dimensionHidden // 2, dimensionHidden // 4).to(device)
        self._hiddenToPrediction3 = torch.nn.Linear(dimensionHidden // 4, dimensionOut).to(device)

        self._batchSize = -1

    def resetSequence(self, batchSize):
        # seems to be costly but is needed
        # otherwise, backward() complains
        self._batchSize = batchSize
        self._hidden = (torch.zeros(self._numLayers, self._batchSize, self._dimensionHidden).to(self._device),
                        torch.zeros(self._numLayers, self._batchSize, self._dimensionHidden).to(self._device))

    def forward(self, x):
        # essayons de passer un PackedSequence

        # evaluate LSTM module
        out, self._hidden = self._lstm(x, self._hidden)

        # out.shape = (batch, sequence_len, self._dimensionHidden)
        # only keep last element of sequence dimension since this
        # is our desired prediction
        out = out[:, -1, :]

        # remap out so it can be fed into self._hiddenToPrediction
        out = out.view(self._batchSize, -1)

        out = torch.nn.functional.relu(self._hiddenToPrediction1(out))
        out = torch.nn.functional.relu(self._hiddenToPrediction2(out))
        out = self._hiddenToPrediction3(out)
        return out

    def parameterCount(self):
        return np.sum(p.numel() for p in self.parameters() if p.requires_grad)
