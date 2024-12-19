# (1) 필요한 라이브러리 임포트
import pandas as pd           # 판다스: CSV 등 파일로부터 데이터를 다루는 데 사용
import numpy as np            # 넘파이: 수치 계산용
from sklearn.model_selection import train_test_split  # 학습/테스트 데이터 나누기 위한 함수
from sklearn.linear_model import LinearRegression     # 선형회귀 모델
from sklearn.preprocessing import OneHotEncoder       # 범주형 문자 데이터를 수치화하기 위한 인코더
from sklearn.metrics import mean_absolute_error, mean_squared_error  # 모델 평가를 위한 함수
from datetime import datetime

# (2) CSV 데이터 불러오기
# real_estate_data.csv 파일은 area, neighborhood, date, price 컬럼을 가진다고 가정합니다.
data = pd.read_csv('real_estate_data.csv')

# 데이터 샘플을 확인 (상위 5행)
print("데이터 상위 5행 출력:")
print(data.head())

# (3) 데이터 전처리

# 3-1. 거래일자(date) 처리
# date는 예를 들어 "2021-05-10" 이런 형태라고 가정합니다.
# 날짜 문자열을 연도, 월, 일 형태의 숫자로 변환합니다.
# 이유: 모델은 일반적으로 날짜 형태 그 자체를 이해하지 못하므로, 연도(year), 월(month) 등의 숫자 형태로 변환하여
#      시간 정보의 패턴을 반영하게 합니다.
data['date'] = pd.to_datetime(data['date']) # 문자열 -> datetime 타입으로 변환
data['year'] = data['date'].dt.year        # 연도 추출
data['month'] = data['date'].dt.month      # 월 추출
data['day'] = data['date'].dt.day          # 일 추출

# date 컬럼은 이제 숫자로 변환한 year, month, day로 대체할 수 있으므로 원본 date 컬럼은 제거
data = data.drop('date', axis=1)

# 3-2. 범주형 변수(neighborhood) 처리
# 동네는 문자열이므로, 이 값을 모델이 이해할 수 있는 숫자 형태로 변환해야 합니다.
# One-Hot Encoding을 사용하면, 예를 들어 neighborhood 컬럼에 "Gangnam", "Jongno"가 있다면
# "neighborhood_Gangnam", "neighborhood_Jongno" 같은 형태의 0/1 컬럼을 만들어줍니다.
#
# OneHotEncoder 사용시 주의:
# - scikit-learn의 OneHotEncoder는 데이터프레임이 아닌 넘파이 배열을 다룹니다.
# - 컬럼을 인코딩한 뒤 데이터프레임으로 다시 합치는 과정이 필요합니다.

# 우선, 인코더를 생성
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# 인코더에 neighborhood 컬럼을 fit_transform
# fit_transform: 인코더가 데이터의 형태(고유한 동네 이름들)를 학습하고, 바로 변환까지 수행
neighborhood_encoded = encoder.fit_transform(data[['neighborhood']]) 

# 인코딩된 컬럼(원-핫 인코딩 결과)을 데이터프레임으로 변환
# get_feature_names_out(): 인코딩된 컬럼 이름을 얻을 수 있음
encoded_cols = pd.DataFrame(neighborhood_encoded, columns=encoder.get_feature_names_out(['neighborhood']))

# 기존 데이터프레임과 합치고, 기존 neighborhood 컬럼 삭제
data = pd.concat([data.drop('neighborhood', axis=1), encoded_cols], axis=1)

# (4) 입력(X)과 타겟(y) 나누기
# 우리가 예측해야 하는 값: price (가격)
y = data['price']   # 목표값
X = data.drop('price', axis=1) # 입력값(특징값)

# (5) 학습 데이터와 테스트 데이터 분리
# train_test_split 함수를 사용해서 데이터를 섞은 뒤 80%는 학습용, 20%는 테스트용으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (6) 모델 선택 및 학습
# 여기서는 간단히 선형회귀 모델을 사용합니다.
model = LinearRegression()

# 모델 학습
model.fit(X_train, y_train)

# (7) 모델 평가
# 테스트 데이터로 예측을 수행
y_pred = model.predict(X_test)

# 평가 지표로 평균절대오차(MAE), 평균제곱오차(MSE), 그리고 루트평균제곱오차(RMSE)를 사용해볼 수 있음
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n모델 평가 결과:")
print("MAE(평균절대오차):", mae)
print("MSE(평균제곱오차):", mse)
print("RMSE(제곱근평균오차):", rmse)

# (8) 모델 사용 예시
# 새로운 부동산 정보가 있을 때 가격을 예측해볼 수 있습니다.
# 예를 들어,
# 면적: 85, 동네: "Jongno", 거래일자: "2023-02-15"
# 이라고 할 때, 이 데이터를 모델에 넣으려면 동일한 전처리 과정을 거쳐야 합니다.

# 새 데이터 예시
new_data = pd.DataFrame({
    'area': [85],
    'neighborhood': ['Jongno'],
    'year': [2023],   # 추후 변환을 위해 직접 할 수도 있지만 여기서는 바로 연,월,일을 기입한다고 가정
    'month': [2],
    'day': [15]
})

# 새 데이터에 대한 neighborhood 인코딩
# fit 시점은 기존 데이터로 완료했으므로, 새로운 데이터에 대해서는 transform 만 수행
new_neighborhood_encoded = encoder.transform(new_data[['neighborhood']])
new_encoded_cols = pd.DataFrame(new_neighborhood_encoded, columns=encoder.get_feature_names_out(['neighborhood']))

# neighborhood 컬럼 제거 후 인코딩한 컬럼 붙이기
new_data = pd.concat([new_data.drop('neighborhood', axis=1), new_encoded_cols], axis=1)

# 새로운 데이터의 예측 수행
new_pred = model.predict(new_data)
print("\n새로운 데이터의 예측 결과 (면적:85, 동네:Jongno, 날짜:2023/02/15):", new_pred[0])
