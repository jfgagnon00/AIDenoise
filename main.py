import data
import model
import numpy as np
import torch


# for reproductability
np.random.seed(0)

# parameters for neural network learning
EPOCHS = 32
BATCH_SIZE = 32
LEARNING_RATE = 0.001

dataset = data.Dataset("./data/filtered_positions.csv")
model = model.FilterModel(1, 10)
print(f"Num parameters: {model.parameterCount()}")

# pytorch objects needed for traning
dataLoader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
lossFunction = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training per say
iteration = 0
model.train()
for epoch in range(EPOCHS):
    for batch, (x, target) in enumerate(dataLoader):
        output = model(x)
        loss = lossFunction(output, target)

        if iteration % 100 == 0:
            print(f"iteration: {iteration:5} loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1

print("Training done")