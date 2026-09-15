import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# --------------------------------
# 1. 데이터 불러오기
# --------------------------------

df = pd.read_csv(
    "data/processed/clean_mart.csv"
)

print("전체 데이터 크기:")
print(df.shape)


# --------------------------------
# 2. 사용할 변수
# --------------------------------

numeric_features = [
    "총_유동인구_수",
    "2030비율",
    "점포_수",
    "폐업비율"
]

categorical_features = [
    "서비스_업종_코드_명",
    "상권_코드_명"
]

target = "점포당_월매출"


# --------------------------------
# 3. 필요한 데이터만 선택
# --------------------------------

model_df = df[
    [
        "기준_년_코드",
        "기준_분기_코드"
    ]
    + numeric_features
    + categorical_features
    + [target]
].copy()

model_df = model_df.dropna()


# --------------------------------
# 4. 2025년 데이터만 사용
# --------------------------------

model_df = model_df[
    model_df["기준_년_코드"] == 2025
].copy()

print("\n2025년 데이터:")
print(model_df.shape)


# --------------------------------
# 5. 상위 1% 이상치 제거
# --------------------------------

upper_limit = model_df[target].quantile(0.99)

model_df = model_df[
    model_df[target] <= upper_limit
].copy()

print("\n이상치 제거 후:")
print(model_df.shape)


# --------------------------------
# 6. 학습 / 테스트 분리
# --------------------------------

train_df = model_df[
    model_df["기준_분기_코드"].isin([1, 2, 3])
].copy()

test_df = model_df[
    model_df["기준_분기_코드"] == 4
].copy()


print("\n학습 데이터 (1~3분기):")
print(train_df.shape)

print("\n테스트 데이터 (4분기):")
print(test_df.shape)


# --------------------------------
# 7. X, y 만들기
# --------------------------------

feature_columns = (
    numeric_features
    + categorical_features
)

X_train = train_df[
    feature_columns
]

y_train = train_df[target]

X_test = test_df[
    feature_columns
]

y_test = test_df[target]


# --------------------------------
# 8. 문자 데이터 전처리
# --------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "category",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# --------------------------------
# 9. RandomForest
# --------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)


# --------------------------------
# 10. 파이프라인 만들기
# --------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# --------------------------------
# 11. 모델 학습
# --------------------------------

print("\n시간 기준 모델 학습 시작...")

pipeline.fit(
    X_train,
    y_train
)

print("시간 기준 모델 학습 완료!")


# --------------------------------
# 12. 4분기 예측
# --------------------------------

pred = pipeline.predict(
    X_test
)


# --------------------------------
# 13. 성능 평가
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
print("2025 Q4 미래 예측 평가")
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
# 14. 예측 결과 샘플
# --------------------------------

result = test_df[
    [
        "상권_코드_명",
        "서비스_업종_코드_명",
        target
    ]
].copy()

result["AI_예상_월매출"] = pred

result["예측오차"] = (
    result[target]
    - result["AI_예상_월매출"]
).abs()


result = result.sort_values(
    "예측오차"
)


print("\n예측 결과 샘플:")
print(
    result.head(20)
)


# --------------------------------
# 15. 결과 저장
# --------------------------------

result.to_csv(
    "data/processed/q4_prediction_result.csv",
    index=False,
    encoding="utf-8-sig"
)


# --------------------------------
# 16. 모델 저장
# --------------------------------

joblib.dump(
    pipeline,
    "models/random_forest_time_model.pkl"
)


print("\n결과 저장 완료:")
print(
    "data/processed/q4_prediction_result.csv"
)

print("\n모델 저장 완료:")
print(
    "models/random_forest_time_model.pkl"
)