import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def main():
    (BASE_DIR / "data/processed").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "models").mkdir(exist_ok=True)
    # 통합 데이터 불러오기

    df = pd.read_csv(
        BASE_DIR / 'data/processed/final_mart.csv'
    )

    print("=== 데이터 불러오기 완료 ===")
    print(df.shape)

    # 전체 열 이름 확인

    print("\n=== 전체 열 이름 ===")
    print(df.columns.tolist())

    # 결측치 확인

    print("\n=== 결측치 개수 ===")

    missing = df.isna().sum()

    print(
        missing[
            missing > 0
        ].sort_values(ascending=False)
    )

    # 핵심 분석 데이터 만들기

    clean_df = df.dropna(
        subset=[
            "점포당_월매출",
            "총_유동인구_수",
            "2030비율"
        ]
    ).copy()

    # 이상값 확인을 위한 기본 통계

    print("\n=== 주요 숫자 데이터 통계 ===")

    print(
        clean_df[
            [
                "분기당_매출_금액",
                "분기당_매출_건수",
                "점포_수",
                "객단가",
                "점포당_월매출",
                "폐업비율",
                "총_유동인구_수",
                "2030비율"
            ]
        ].describe()
    )

    # 정제 전 / 후 데이터 크기 비교

    print("\n=== 데이터 크기 비교 ===")

    print("정제 전:", df.shape)
    print("정제 후:", clean_df.shape)

    # 정제 데이터 저장

    clean_df.to_csv(
        BASE_DIR / 'data/processed/clean_mart.csv',
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== 저장 완료 ===")
    print("data/processed/clean_mart.csv")


if __name__ == "__main__":
    main()
