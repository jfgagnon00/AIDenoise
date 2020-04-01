import data
import model
import numpy as np
import torch


# for reproductability
np.random.seed(0)

EPOCHS = 32
BATCH_SIZE = 32
LEARNING_RATE = 0.001

dataset = data.Dataset("./data/filtered_positions.csv")
model = model.FilterModel(1, 10)
print(f"Num parameters: {model.parameterCount()}")

dataLoader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
lossFunction = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.train()

for epoch in range(EPOCHS):
    for batch, (x, target) in enumerate(dataLoader):
        output = model(x)
        loss = lossFunction(output, target)

        if batch % 100 == 0:
            print(f"epoch: {epoch} batch: {batch}, loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Done")