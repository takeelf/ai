from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from joblib import dump
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder

client = MongoClient("mongodb+srv://takeelf:dnjsgh11!!aA@real-estate-cluster.sx265.mongodb.net/?retryWrites=true&w=majority&appName=real-estate-cluster")
db = client['real_estate']
collection = db['transaction_price']

pipeline = [
    {
        "$project": {
            "_id": 0,  # _id 필드 제외
            "dealYear": 1,
            "dealMonth": 1,
            "excluUseAr": 1,
            "umdNm": 1,
            "dealAmount": 1
        }
    }
]

cursor = collection.aggregate(pipeline)
data = list(cursor)

df = pd.DataFrame(data)
df = df.dropna()

label_encoder = LabelEncoder()
df['umdNm'] = label_encoder.fit_transform(df['umdNm'])

dump(label_encoder, 'realestate_label_encoder.pkl')

# X(features)와 y(target) 분리
X = df.drop('dealAmount', axis=1)  # target 제외한 모든 컬럼
y = df['dealAmount']     

client.close()

model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = torch.FloatTensor(X_scaled)
Y = torch.FloatTensor(y).reshape(-1, 1)

batch_size = 32
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
        
    if epoch % 2000 == 0:
        print(f"2000 Epoch {epoch}, Loss: {loss.item()}")

# 학습이 끝난 후 모델 저장
MODEL_PATH = 'realestate_model.pth'
SCALER_PATH = 'realestate_scaler.pkl'
dump(scaler, SCALER_PATH)

# 모델 상태 저장
torch.save(model, MODEL_PATH)

print(f"모델이 저장되었습니다: {MODEL_PATH}")
print(f"Scaler가 저장되었습니다: {SCALER_PATH}")
