from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
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
client.close()

df = pd.DataFrame(data)
df = df.dropna()

print(f"이촌동 데이터 개수: {len(df)}")

label_encoder = LabelEncoder()
df['umdNm'] = label_encoder.fit_transform(df['umdNm'])

dump(label_encoder, 'realestate_rf_label_encoder.pkl')

# X(features)와 y(target) 분리
x = df.drop('dealAmount', axis=1)  # target 제외한 모든 컬럼
y = df['dealAmount']     

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1  # 병렬 처리
)

# scaler = StandardScaler()
# target_scaler = StandardScaler()
# X_scaled = scaler.fit_transform(x)
# y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

# 학습이 끝난 후 모델 저장
MODEL_PATH = 'realestate_rf_model.pth'
SCALER_X_PATH = 'realestate_rf_scaler_x.pkl'
SCALER_Y_PATH = 'realestate_rf_scaler_y.pkl'

model.fit(x, y)

# 모델 상태 저장
torch.save(model, MODEL_PATH)
# dump(scaler, SCALER_X_PATH)
# dump(target_scaler, SCALER_Y_PATH)

print(f"모델이 저장되었습니다: {MODEL_PATH}")
# print(f"Scaler가 저장되었습니다: {SCALER_X_PATH}")
