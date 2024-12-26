import torch
import joblib
import numpy as np
import pandas as pd

model = torch.load('realestate_model.pth')
scaler_x = joblib.load('realestate_scaler_x.pkl')
scaler_y = joblib.load('realestate_scaler_y.pkl')
location_encoder = joblib.load('realestate_label_encoder.pkl')

feature_names = ['dealMonth', 'dealYear', 'excluUseAr', 'umdNm']

# 2. 실제 환자 데이터 입력 예시
real_estate_data = [
    1,        # dealMonth (거래월)
    2006,         # dealYear (거래년도)
    60,      # excluUseAr (전용면적)
    '이촌동',     # umdNm (동이름)
]

patient_df = pd.DataFrame([real_estate_data], columns=feature_names)
patient_df['umdNm'] = location_encoder.transform(patient_df['umdNm'])

scaled_data = scaler_x.transform(patient_df)
input_tensor = torch.FloatTensor(scaled_data)

model.eval()

with torch.no_grad():
    prediction = model(input_tensor)
    prediction_reshaped = prediction.numpy().reshape(-1, 1)  # shape를 (1, 1)로 조정
    
    real_price = scaler_y.inverse_transform(prediction_reshaped)
    
    print(f"예상 거래금액: {real_price[0][0]:.2f}만원")

