from sklearn.datasets import load_diabetes
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

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

X = torch.FloatTensor(df.iloc[:, :10].values)
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
        
    if epoch % 2000 == 0:
        print(f"2000 Epoch {epoch}, Loss: {loss.item()}")

# 학습 과정이 모두 끝난 뒤에 예측을 하는 예시

# 1. 기존 데이터로 예측하기
model.eval()  # 추론 모드로 전환 (dropout, batchnorm 등이 있을 경우를 대비)
with torch.no_grad():
    predictions = model(X)  # X는 기존 우리가 이용한 입력 데이터

# 예측 값과 실제 값 비교
print("Predictions on training set:")
print(predictions[:10])  # 처음 10개 샘플의 예측 결과
print("Actual values:")
print(Y[:10])  # 실제 타겟 값

# 2. 새로운 샘플로 예측하기
# 당뇨병 데이터셋의 각 특성 순서는 dataset.feature_names로 확인 가능
# ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"] 
# 예를 들어, 새로운 환자의 특성을 다음과 같이 가정한다면:
new_sample = torch.FloatTensor([[ 0.03, -0.05, 0.06, 0.02, -0.04, -0.02, 0.05, 0.04, -0.01, 0.03 ]])

model.eval()
with torch.no_grad():
    new_pred = model(new_sample)

print(f"New sample prediction: {new_pred.item()}")
