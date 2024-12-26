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
        "$match": {
            "umdNm": "이촌동"  # 이촌동 데이터만 필터링
        }
    },
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

print(f"이촌동 데이터 개수: {len(df)}")

label_encoder = LabelEncoder()
df['umdNm'] = label_encoder.fit_transform(df['umdNm'])

# plt.figure(figsize=(10, 6))
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.scatter(df.iloc[:, i], df['dealAmount'])
#     plt.title(df.columns[i])

# plt.tight_layout()
# plt.show()


dump(label_encoder, 'realestate/realestate_label_encoder.pkl')

# X(features)와 y(target) 분리
x = df.drop('dealAmount', axis=1)  # target 제외한 모든 컬럼
y = df['dealAmount']     

client.close()

model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

scaler = StandardScaler()
target_scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
X = torch.FloatTensor(X_scaled)
y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1))
Y = torch.FloatTensor(y_scaled)

# 학습이 끝난 후 모델 저장
MODEL_PATH = 'realestate/realestate_model.pth'
SCALER_X_PATH = 'realestate/realestate_scaler_x.pkl'
SCALER_Y_PATH = 'realestate/realestate_scaler_y.pkl'

batch_size = 64
learning_rate = 1e-4
epochs = 20000

optimizer = torch.optim.Adam(params = model.parameters(), lr=learning_rate)

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
        
        # print(f"each Epoch {epoch}, Loss: {loss.item()}")
      
    if epoch % 10 == 0:
        print(f"10 Epoch {epoch}, Loss: {loss.item()}")


# 모델 상태 저장
torch.save(model, MODEL_PATH)
dump(scaler, SCALER_X_PATH)
dump(target_scaler, SCALER_Y_PATH)

print(f"모델이 저장되었습니다: {MODEL_PATH}")
print(f"Scaler가 저장되었습니다: {SCALER_X_PATH}")
