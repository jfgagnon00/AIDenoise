import data
import itertools
import model
import numpy as np
import torch


# for reproductability
np.random.seed(0)
torch.manual_seed(0)

# parameters for neural network learning
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0025

LSTM_SEQUENCE_LENGTH = 40
LSTM_NUM_LAYERS = 1
LSTM_HIDDEN_LAYER_SIZE = LSTM_SEQUENCE_LENGTH

dataset = data.Dataset("./data/filtered_positions.csv", LSTM_SEQUENCE_LENGTH)
model = model.FilterModel(dimensionHidden=LSTM_HIDDEN_LAYER_SIZE, numLayers=LSTM_NUM_LAYERS, batchSize=BATCH_SIZE)
print(f"Num parameters: {model.parameterCount()}")

# pytorch objects needed for traning
dataLoader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
lossFunction = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training per say
iteration = 0
model.train()
for epoch in range(EPOCHS):
    for batch, (x, target) in enumerate(dataLoader):
        model.resetSequence(x.shape[0])
        output = model(x)

        loss = lossFunction(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(f"iteration: {iteration:4} (epoch: {epoch:3}, batch: {batch:3}) loss: {loss.item():2.8f}")

        iteration += 1

print("Training done")

print("Testing")
with open("./data/new_positions.csv", "w") as output:
    output.write("CartID,Index,RawPos,MovingAverage,LSTMPos\n")
    index = 2

    rawPositions = dataset.rawPositions
    rawPositions = rawPositions.reshape(rawPositions.shape[0], 1)
    rawPositions = torch.from_numpy(rawPositions)

    model.eval()
    model.resetSequence(rawPositions.shape[0])
    filteredPositions = model(rawPositions)

    def remap(value):
        return (value - dataset.bias) / dataset.scale

    for cartID, \
        rawPosition, \
        orignalFilter, \
        filteredPosition in \
        itertools.zip_longest(dataset.cartIDs,
                              dataset.rawPositions,
                              dataset.filteredPositions,
                              filteredPositions):
        output.write(f"{int(cartID)}, {int(index)}, {remap(rawPosition)}, {remap(orignalFilter)}, {remap(filteredPosition.item())}\n")
        index += 1

print("Testing done")
