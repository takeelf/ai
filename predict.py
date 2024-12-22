import torch
import joblib
import numpy as np
import pandas as pd

model = torch.load('diabetes_model.pth')
scaler = joblib.load('diabetes_scaler.pkl')
model.eval()

feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# 2. 실제 환자 데이터 입력 예시
real_patient_data = [
    59,        # age (나이)
    2,         # sex (성별)
    32.1,      # bmi (체질량지수)
    101.0,     # bp (평균 혈압)
    157,       # s1 (총 콜레스테롤)
    93.2,      # s2 (저밀도 지단백)
    38,        # s3 (고밀도 지단백)
    4,         # s4 (총 콜레스테롤/HDL)
    4.8598,    # s5 (로그 혈청 트리글리세리드)
    87         # s6 (혈당)
]

patient_df = pd.DataFrame([real_patient_data], columns=feature_names)

scaled_data = scaler.transform(patient_df)
input_tensor = torch.FloatTensor(scaled_data)

with torch.no_grad():
    prediction = model(input_tensor)

print(f"당뇨병 진행도 예측값: {prediction.item():.2f}")