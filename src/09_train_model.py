import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import joblib


# --------------------------------
# 1. 데이터 불러오기
# --------------------------------

df = pd.read_csv(
    "data/processed/clean_mart.csv"
)

print("데이터 크기:")
print(df.shape)


# --------------------------------
# 2. 사용할 컬럼 선택
# --------------------------------

features = [
    "총_유동인구_수",
    "2030비율",
    "점포_수",
    "폐업비율"
]

target = "점포당_월매출"


# --------------------------------
# 3. 필요한 데이터만 선택
# --------------------------------

model_df = df[
    features + [target]
].copy()


# 결측치 제거
model_df = model_df.dropna()


print("\n모델 학습 데이터 크기:")
print(model_df.shape)


# --------------------------------
# 4. X, y 나누기
# --------------------------------

X = model_df[features]

y = model_df[target]


# --------------------------------
# 5. 학습용 / 테스트용 데이터 분리
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n학습 데이터:")
print(X_train.shape)

print("테스트 데이터:")
print(X_test.shape)


# --------------------------------
# 6. RandomForest 모델 만들기
# --------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# --------------------------------
# 7. 모델 학습
# --------------------------------

print("\n모델 학습 시작...")

model.fit(
    X_train,
    y_train
)

print("모델 학습 완료!")


# --------------------------------
# 8. 예측
# --------------------------------

pred = model.predict(
    X_test
)


# --------------------------------
# 9. 성능 평가
# --------------------------------

mae = mean_absolute_error(
    y_test,
    pred
)

rmse = mean_squared_error(
    y_test,
    pred
) ** 0.5

r2 = r2_score(
    y_test,
    pred
)


print("\n============================")
print("모델 평가 결과")
print("============================")

print(
    f"MAE : {mae:,.0f}원"
)

print(
    f"RMSE : {rmse:,.0f}원"
)

print(
    f"R² : {r2:.4f}"
)


# --------------------------------
# 10. 변수 중요도 확인
# --------------------------------

importance = pd.DataFrame(
    {
        "변수": features,
        "중요도": model.feature_importances_
    }
)

importance = importance.sort_values(
    "중요도",
    ascending=False
)

print("\n변수 중요도:")
print(importance)


# --------------------------------
# 11. 모델 저장
# --------------------------------

joblib.dump(
    model,
    "models/random_forest_sales.pkl"
)

print(
    "\n모델 저장 완료:"
)

print(
    "models/random_forest_sales.pkl"
)