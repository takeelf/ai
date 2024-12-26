from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd

class IrisDataset(Dataset):
    def __init__(self, train=True):
        dataset = load_iris()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, test_size=0.3, random_state=827
        )        
        
        if train:
            self.data = torch.FloatTensor(x_train)
            self.target = torch.LongTensor(y_train)
        else:
            self.data = torch.FloatTensor(x_test)
            self.target = torch.LongTensor(y_test)
            
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    
    def __len__(self):
        return len(self.data)

batch_size = 64
learning_rate = 1e-3
epochs = 2000

model = nn.Sequential(
    nn.Linear(4, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)

train_dataset = IrisDataset(train=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for data, target in train_dataloader:
        optimizer.zero_grad()
        
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        
test_dataset = IrisDataset(train=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

num_correct = 0

with torch.no_grad():
    for data, target in test_dataloader:
        outputs = model(data)
        pred = torch.max(outputs, 1)[1]
        corr = pred.eq(target).sum().item()
        num_correct += corr
    
    print(f"Accuracy: {num_correct / len(test_dataset) * 100:.2f}%")