import data
import itertools
import model
import numpy as np
import time
import torch


# parameters for neural network learning
EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 0.001

LSTM_SEQUENCE_LENGTH = 40
LSTM_HIDDEN_LAYER_SIZE = 64

NUM_WORKERS = 0  # 4

dataset = data.Dataset([
    "./data/filtered_positions_00.csv",
    "./data/filtered_positions_01.csv",
    "./data/filtered_positions_02.csv",
    "./data/filtered_positions_03.csv"
    ])


def train(model, device):
    print("Training")

    # prepare dataset for the loader
    dataset.preprocess(LSTM_SEQUENCE_LENGTH, device)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # pytorch objects needed for traning
    lossFunction = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    iteration = 0
    model.train()
    timeBegin = time.perf_counter()
    for epoch in range(EPOCHS):
        # build a pack sequence of batches
        # dramatically improve performance of training
        featureList = []
        targetList = []
        for features, target in dataLoader:
            features = features.permute(1, 0, 2)
            featureList.append(features)
            targetList.append(target)

        # shape is (num_batches, BATCH_SIZE, LSTM_SEQUENCE_LENGTH, dataset.numFeatures())
        featureList = torch.nn.utils.rnn.pack_sequence(featureList)
        targetList = torch.nn.utils.rnn.pack_sequence(targetList)

        # comment on reset la sequence entre les astie de batch!
        print(featureList.batch_sizes.shape)
        print(featureList.batch_sizes)
        model.resetSequence(featureList.batch_sizes[0].item())

        optimizer.zero_grad()
        output = model(featureList)
        loss = lossFunction(output, targetList)
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(f"iteration: {iteration:4} (epoch: {epoch:3} batch: {batch:4}) ")  # loss: {loss.item():2.8f}

        iteration += 1
    timeEnd = time.perf_counter()
    print(f"Training done: {timeEnd - timeBegin:.3f}s")


def filter(dataset, model, device):
    return

    print("Filter")

    with open("./data/new_positions.csv", "w") as output:
        output.write("CartID,Index,RawPos,MovingAverage,LSTMPos\n")

        rawPositions = dataset.rawPositions
        rawPositions = rawPositions.reshape(rawPositions.shape[0], 1)
        rawPositions = torch.from_numpy(rawPositions).to(device)

        model.eval()
        model.resetSequence(rawPositions.shape[0])
        filteredPositions = model(rawPositions).cpu()

        def remap(value):
            return (value - dataset.bias) / dataset.scale

        index = 2
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

        device = torch.device("cuda:0" if cudaAvailable else "cpu")
        model = model.FilterModel(device, dimensionIn=dataset.numFeatures(), dimensionHidden=LSTM_HIDDEN_LAYER_SIZE)
        print(f"Num parameters: {model.parameterCount()}")

        train(model, device)
        filter(dataset, model, device)
    else:
        print("Worker")
