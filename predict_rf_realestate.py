import torch
import joblib
import numpy as np
import pandas as pd

model = torch.load('realestate_rf_model.pth')
# scaler_x = joblib.load('realestate_rf_scaler_x.pkl')
# scaler_y = joblib.load('realestate_rf_scaler_y.pkl')
location_encoder = joblib.load('realestate_rf_label_encoder.pkl')

feature_names = ['dealMonth', 'dealYear', 'excluUseAr', 'umdNm']

# 2. 실제 환자 데이터 입력 예시
real_estate_data = [
    12,        # dealMonth (거래월)
    2006,         # dealYear (거래년도)
    90,      # excluUseAr (전용면적)
    '이촌동',     # umdNm (동이름)
]

input_df = pd.DataFrame([real_estate_data], columns=feature_names)
input_df['umdNm'] = location_encoder.transform(input_df['umdNm'])

# scaled_data = scaler_x.transform(input_df)

# 4. 예측
prediction = model.predict(input_df)

print(f"예상 거래금액(예측값): {prediction[0]:.2f} 만원")

