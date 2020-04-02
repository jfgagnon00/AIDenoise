import data
import itertools
import model
import numpy as np
import torch


# parameters for neural network learning
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.002

LSTM_SEQUENCE_LENGTH = 40
LSTM_NUM_LAYERS = 1
LSTM_HIDDEN_LAYER_SIZE = 16  # LSTM_SEQUENCE_LENGTH

dataset = data.Dataset("./data/filtered_positions.csv", LSTM_SEQUENCE_LENGTH)


def train(dataloader, model, device):
    print("Training")

    # pytorch objects needed for traning
    lossFunction = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training per say
    iteration = 0
    model.train()
    for epoch in range(EPOCHS):
        for batch, (x, target) in enumerate(dataLoader):
            x = x.to(device)
            target = target.to(device)

            model.resetSequence(x.shape[0])
            optimizer.zero_grad()
            output = model(x)
            loss = lossFunction(output, target)
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print(f"iteration: {iteration:4} (epoch: {epoch:3}, batch: {batch:3}) loss: {loss.item():2.8f}")

            iteration += 1

    print("Training done")


def filter(dataset, model, device):
    print("Filter")
    with open("./data/new_positions.csv", "w") as output:
        output.write("CartID,Index,RawPos,MovingAverage,LSTMPos\n")
        index = 2

        rawPositions = dataset.rawPositions
        rawPositions = rawPositions.reshape(rawPositions.shape[0], 1)
        rawPositions = torch.from_numpy(rawPositions).to(device)

        model.eval()
        model.resetSequence(rawPositions.shape[0])
        filteredPositions = model(rawPositions).cpu()

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

    print("Filter done")


if __name__ == "__main__":
    info = torch.utils.data.get_worker_info()

    if info is None:
        print("Main")

        cudaAvailable = torch.cuda.is_available()
        print(f"Cuda available: {cudaAvailable}")

        # for reproductability
        # not ok with multi process
        np.random.seed(0)
        torch.manual_seed(0)

        dataLoader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        device = torch.device("cuda:0" if cudaAvailable else "cpu")
        model = model.FilterModel(device, dimensionHidden=LSTM_HIDDEN_LAYER_SIZE, numLayers=LSTM_NUM_LAYERS, batchSize=BATCH_SIZE)
        print(f"Num parameters: {model.parameterCount()}")

        train(dataLoader, model, device)
        filter(dataset, model, device)
    else:
        print("Worker")
