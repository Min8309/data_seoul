import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def main():
    (BASE_DIR / "data/processed").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "models").mkdir(exist_ok=True)
    # 데이터 불러오기

    df = pd.read_csv(
        BASE_DIR / 'data/processed/clean_mart.csv'
    )

    print("데이터 크기:")
    print(df.shape)

    # 사용할 변수

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

    # 필요한 데이터 선택

    model_df = df[
        numeric_features
        + categorical_features
        + [target]
    ].copy()

    model_df = model_df.dropna()

    print("\n모델 학습 데이터 크기:")
    print(model_df.shape)

    # X, y 나누기

    X = model_df[
        numeric_features
        + categorical_features
    ]

    y = model_df[target]

    # 학습 / 테스트 분리

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Fit the outlier threshold on training data only; evaluate all test rows.
    upper_limit = y_train.quantile(0.99)
    keep = y_train <= upper_limit
    X_train = X_train.loc[keep]
    y_train = y_train.loc[keep]

    # 문자 데이터 One-Hot Encoding

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

    # RandomForest

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    # 파이프라인

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

    # 학습

    print("\n개선 모델 학습 시작...")

    pipeline.fit(
        X_train,
        y_train
    )

    print("개선 모델 학습 완료!")

    # 예측

    pred = pipeline.predict(
        X_test
    )

    # 평가

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
    print("개선 모델 평가 결과")
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

    # 실제값 / 예측값 비교

    result = pd.DataFrame(
        {
            "실제_월매출": y_test,
            "예측_월매출": pred
        }
    )

    result["오차"] = (
        result["실제_월매출"]
        - result["예측_월매출"]
    ).abs()

    print("\n예측 결과 샘플:")
    print(
        result.head(10)
    )

    # 모델 저장

    joblib.dump(
        pipeline,
        BASE_DIR / 'models/random_forest_sales_v2.pkl'
    )

    print(
        "\n개선 모델 저장 완료:"
    )

    print(
        "models/random_forest_sales_v2.pkl"
    )


if __name__ == "__main__":
    main()
