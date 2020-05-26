import data
import matplotlib.pyplot as plt
import model
import os
import numpy as np
import time
import torch
from torch.utils.tensorboard import SummaryWriter


# general parameters for training
EPOCHS = 300  # 300
LEARNING_RATE = 5e-4  ## 6e-5
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 256
LOG_EVERY = 1
SAVE_EVERY = 10
RELOAD_PREVIOUS_SESSION = True

# parameters for dataset/dataloader
SEQUENCE_LENGTH = 20
NUM_WORKERS = 0

# keys to access training stats
TRAIN_LOSS_KEY = "trainLoss"
TEST_LOSS_KEY = "testLoss"
LEARNING_RATE_KEY = "learningRate"


def createModel(device, datasetClass):
    # model_ = model.BasicRNN(device,
    #                         SEQUENCE_LENGTH,
    #                         1,
    #                         datasetClass.NumFeatures)
    # sequenceLengthOut = 1

    # model_ = model.SimpleLinear(device,
    #                             SEQUENCE_LENGTH,
    #                             datasetClass.NumFeatures)
    # sequenceLengthOut = 1


    # LSTM_HIDDEN_LAYER_SIZE = 10
    # model_ = model.LSTMFilter(device,
    #                           datasetClass.NumFeatures,
    #                           LSTM_HIDDEN_LAYER_SIZE)
    # sequenceLengthOut = SEQUENCE_LENGTH

    ENCODER_CONV_LAYER_SIZES = [10, 5]
    ENCODER_CONV_KERNEL_SIZES = [5, 3]
    model_ = model.ConvAutoEncoderFilter(device,
                                         SEQUENCE_LENGTH,
                                         datasetClass.NumFeatures,
                                         ENCODER_CONV_LAYER_SIZES,
                                         ENCODER_CONV_KERNEL_SIZES)
    sequenceLengthOut = 1

    # DENSE_LAYER_SIZES = [10, 5, 3, 3, 5, 10, SEQUENCE_LENGTH // 2]
    # DENSE_LAYER_SIZES = [10, 10]
    # DENSE_LAYER_SIZES = [20, 10, 5, 2, 5, 10, 20]
    # DENSE_LAYER_SIZES = [10, 10, 1]
    # model_ = model.DenseFilter(device,
    #                            SEQUENCE_LENGTH,
    #                            datasetClass.NumFeatures,
    #                            DENSE_LAYER_SIZES)
    # sequenceLengthOut = DENSE_LAYER_SIZES[-1]

    return model_, sequenceLengthOut


def train(trainDataset, testDataset, model, weightFileName):
    print("Training")

    # pytorch objects needed for training
    lossFunction = torch.nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def lrLambda(epoch):
        return 10.0 ** (epoch / 20.0)
    lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrLambda)

    # training data
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # # testing data
    # testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    outputStatsDict = {
            TRAIN_LOSS_KEY: [],
            TEST_LOSS_KEY: [],
            LEARNING_RATE_KEY: []
        }

    def log(epoch, trainLoss, testLoss, lr):
        print(f"epoch: {epoch:3} - trainLoss: {trainLoss:2.9f}, testLoss: {testLoss:2.9f}, lr: {lr}")

    summaryWriter = SummaryWriter()

    # for i, module in enumerate(model._net):
    #     if hasattr(module, "weight") and module.weight is not None:
    #         summaryWriter.add_histogram(f"L{i}_Weights", module.weight, 0)
    #     if hasattr(module, "bias") and module.bias is not None:
    #         summaryWriter.add_histogram(f"L{i}_Bias", module.bias, 0)
    #     if hasattr(module, "grad") and module.grad is not None:
    #         summaryWriter.add_histogram(f"L{i}_Grad", module.grad, 0)

    for epoch in range(EPOCHS):
        # training
        model.train()
        iteration = 0.0
        sumLoss = 0.0

        model.resetSequence(TRAIN_BATCH_SIZE)
        for x, target in trainDataLoader:
            # model.resetSequence(x.shape[0])
            output = model(x)
            loss = lossFunction(output, target)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            sumLoss += loss.item()
            iteration += 1.0
        # lrScheduler.step()
        trainError = sumLoss / iteration
        outputStatsDict[TRAIN_LOSS_KEY].append(trainError)
        outputStatsDict[LEARNING_RATE_KEY].append(lrScheduler.get_lr())

        if epoch % SAVE_EVERY == 0:
            summaryWriter.add_scalar('trainError', trainError, epoch + 1)
            # for i, module in enumerate(model._net):
            #     if hasattr(module, "weight") and module.weight is not None:
            #         summaryWriter.add_histogram(f"L{i}_Weights", module.weight, epoch + 1)
            #         summaryWriter.add_histogram(f"L{i}_Weights_Grad", module.weight.grad, epoch + 1)
            #     if hasattr(module, "bias") and module.bias is not None:
            #         summaryWriter.add_histogram(f"L{i}_Bias", module.bias, epoch + 1)

        # # testing
        # model.eval()
        # iteration = 0.0
        # sumLoss = 0.0
        # model.resetSequence(TEST_BATCH_SIZE)
        # for x, target, pos in testDataLoader:
        #     output = model(x, pos)
        #     loss = lossFunction(output, target)
        #     sumLoss += loss.item()
        #     iteration += 1.0
        # testError = sumLoss / iteration
        testError = 0.0
        outputStatsDict[TEST_LOSS_KEY].append(testError)

        # keep user informed of progress
        if epoch % LOG_EVERY == 0:
            log(epoch, trainError, testError, lrScheduler.get_lr())

        if epoch % SAVE_EVERY == 0:
            torch.save(model.state_dict(), weightFileName)

    log(epoch, trainError, testError, lrScheduler.get_lr())
    torch.save(model.state_dict(), weightFileName)
    summaryWriter.close()
    print("Training done")

    return outputStatsDict


def validation(dataset, model):
    print("Filter")

    def remap(value):
        return (value - dataset.bias) / dataset.scale

    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()

    with open("./results/new_positions.csv", "w") as output:
        output.write(f"Index,RawPos,MovingAverage,{model.getName()}\n")

        index = 2
        for x, target in dataLoader:
            rawPosition = x[0, -1]
            rawPosition = remap(rawPosition.item())

            originalFilter = target[0, -1]
            originalFilter = remap(originalFilter.item())

            model.resetSequence(1)
            newFilter = model(x)
            newFilter = newFilter[0, -1]
            newFilter = remap(newFilter.item())

            output.write(f"{int(index)}, {rawPosition}, {originalFilter}, {newFilter}\n")
            index += 1

    print("Filter done")


def getModelParameterCount(model):
    generator = [p.numel() for p in model.parameters() if p.requires_grad]
    list = np.fromiter(generator, dtype=np.int)
    return np.sum(list)


if __name__ == "__main__":
    info = torch.utils.data.get_worker_info()

    if info is None:
        print("Main")

        cudaAvailable = torch.cuda.is_available()
        print(f"Cuda available: {cudaAvailable}")

        # for reproductability
        # not ok with multi process
        # np.random.seed(0)
        torch.manual_seed(0)

        device = torch.device("cuda:0" if cudaAvailable else "cpu")
        trainDataset = data.Dataset([
            "./data/filtered_positions_00.csv",
            "./data/filtered_positions_01.csv",
            "./data/filtered_positions_02.csv",
            "./data/filtered_positions_03.csv"
        ])

        model_, sequenceLengthOut = createModel(device, type(trainDataset))
        weightFilename = f"./results/{model_.getName()}.pt"
        print(f"{model_.getName()}, num parameters: {getModelParameterCount(model_)}")

        if RELOAD_PREVIOUS_SESSION and os.path.exists(weightFilename):
            weights = torch.load(weightFilename)
            model_.load_state_dict(weights, strict=False)
            model_.eval()

        trainDataset.applyScaleBias(1.0, 0.0)
        trainDataset.preprocess(device, SEQUENCE_LENGTH, sequenceLengthOut)

        lossHistory = train(trainDataset, None, model_, weightFilename)
        epochs = range(EPOCHS)
        plt.plot(epochs, lossHistory[TRAIN_LOSS_KEY], 'g', label='Training loss')
        # plt.plot(epochs, lossHistory[TEST_LOSS_KEY], 'b', label='Test loss')
        # plt.title('Training and Validation loss')
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("./results/loss.png")

        plt.clf()
        plt.semilogx(lossHistory[LEARNING_RATE_KEY], lossHistory[TRAIN_LOSS_KEY], 'g', label='Loss vs LR')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("./results/lr.png")

        validatinDataset = data.Dataset([
            "./data/filtered_positions_00.csv",
            "./data/filtered_positions_01.csv",
            "./data/filtered_positions_02.csv",
            "./data/filtered_positions_03.csv"
        ])
        validatinDataset.applyScaleBias(1.0, 0.0)
        validatinDataset.preprocess(device, SEQUENCE_LENGTH, sequenceLengthOut)
        validation(validatinDataset, model_)
    else:
        print("Worker")
