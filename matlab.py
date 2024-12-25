from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from joblib import dump

dataset = load_diabetes()
print(dataset.keys())

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

# print(df.head())
# plt.figure(figsize=(20,10))
# for i in range(10):
#     plt.subplot(3, 4, i+1)
#     plt.scatter(df.iloc[:, i], df['target'])
#     plt.title(df.columns[i])

# plt.tight_layout()
# plt.show()

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.iloc[:, :10])
X = torch.FloatTensor(X_scaled)
Y = torch.FloatTensor(df['target'].values).reshape(-1, 1)

batch_size = 64
learning_rate = 1e-4
epochs = 30000

optimizer = torch.optim.SGD(params = model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for i in range(0, len(X), batch_size):
        start = i
        end = min(start + batch_size, len(X))
        
        x = X[start:end]
        y = Y[start:end]
        
        optimizer.zero_grad()
        
        pred_y = model(x)
        loss = nn.MSELoss()(pred_y, y)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        
    if epoch % 2000 == 0:
        print(f"2000 Epoch {epoch}, Loss: {loss.item()}")

# 학습이 끝난 후 모델 저장
MODEL_PATH = 'diabetes_model.pth'
SCALER_PATH = 'diabetes_scaler.pkl'
dump(scaler, SCALER_PATH)

# 모델 상태 저장
torch.save(model, MODEL_PATH)

print(f"모델이 저장되었습니다: {MODEL_PATH}")
print(f"Scaler가 저장되었습니다: {SCALER_PATH}")
