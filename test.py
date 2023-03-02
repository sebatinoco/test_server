from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# cargamos datos
X = load_iris()['data']
y = load_iris()['target']

# transformamos a tensores
X = torch.tensor(X)
y = torch.tensor(y)

train_data = TensorDataset(X, y)

batch_size = 16
train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 32, random_state = 3380):
        
        super(MLP, self).__init__()

        if random_state:
            torch.manual_seed(random_state)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

# HIPERPARÁMETROS
hidden_dim = 256
model = MLP(input_dim = X.shape[1], output_dim = 3, hidden_dim = hidden_dim)
epochs = 16
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = torch.nn.CrossEntropyLoss()

n_batches = len(train_loader)
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    total = 0
    correct = 0
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        
        predictions = torch.argmax(outputs, dim = 1)
        total += len(labels)
        correct += (predictions == labels).sum().item()

        sys.stdout.write(f'\rEpoch: {epoch+1:03d} \t Avg Train Loss: {running_loss/n_batches:.3f} \t Train Accuracy: {100 * correct/total:.2f} %') # print de loss promedio x epoca

