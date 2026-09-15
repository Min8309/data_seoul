import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


# --------------------------------
# 1. 데이터 불러오기
# --------------------------------

df = pd.read_csv(
    "data/processed/clean_mart.csv"
)

print("전체 데이터:")
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
    numeric_features
    + categorical_features
    + [target]
].copy()

model_df = model_df.dropna()

print("\n결측치 제거 후:")
print(model_df.shape)


# --------------------------------
# 4. 상위 1% 이상치 제거
# --------------------------------

upper_limit = model_df[target].quantile(0.99)

model_df = model_df[
    model_df[target] <= upper_limit
].copy()

print("\n상위 1% 이상치 제거 후:")
print(model_df.shape)


# --------------------------------
# 5. X / y
# --------------------------------

X = model_df[
    numeric_features
    + categorical_features
]

y = model_df[target]


# --------------------------------
# 6. 문자 데이터 처리
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
# 7. RandomForest
# --------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)


# --------------------------------
# 8. 파이프라인
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
# 9. 최종 모델 학습
# --------------------------------

print("\n최종 AI 모델 학습 시작...")

pipeline.fit(
    X,
    y
)

print("최종 AI 모델 학습 완료!")


# --------------------------------
# 10. 모델 저장
# --------------------------------

joblib.dump(
    pipeline,
    "models/final_sales_model.pkl"
)


print("\n============================")
print("최종 AI 모델 저장 완료")
print("============================")

print(
    "models/final_sales_model.pkl"
)