import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import time
import json

import torch_model

# read params

params = {}

with open('params.json') as json_file:
    params = json.load(json_file)

# read data
X_train = pd.read_csv(os.path.join('data', 'X_train.csv'), header=None).values
y_train = pd.read_csv(os.path.join('data', 'y_train.csv'), header=None).values

X_test = pd.read_csv(os.path.join('data', 'X_test.csv'), header=None).values
y_test = pd.read_csv(os.path.join('data', 'y_test.csv'), header=None).values

# create dataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        assert (X.shape[0] == y.shape[0])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx]))


trainset = Dataset(X_train, y_train)
testset = Dataset(X_test, y_test)

# create dataloader

batch_size = params['batch_size']

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

# create model

model = torch_model.DNN()

# training

epochs = params['epochs']
learning_rate = params['learning_rate']

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

start = time.time()
for epoch in range(epochs):

    current_loss = 0.0
    current_val_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        current_loss += loss.item()
    current_loss /= len(trainloader)
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            current_val_loss += loss.item()
    current_val_loss /= len(testloader)
    print(
        f'epoch: {epoch}, train loss: {current_loss}, validation loss : {current_val_loss}'
    )

end = time.time()
elapsed = end - start
print(f'time: {elapsed}s')
preds = []
trues = []
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = model(inputs)
        preds.append(outputs.flatten().tolist())
        trues.append(labels.flatten().tolist())

preds = np.array([item for sublist in preds for item in sublist])
true = np.array([item for sublist in trues for item in sublist])

pred = np.array(preds > 0.5, dtype=np.float32)

np.savetxt(os.path.join('data', 'torch_pred.csv'), pred, delimiter=',')
