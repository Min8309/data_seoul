import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def minmax_score(series):
    """Constant inputs receive a neutral score; missing values remain missing."""
    spread = series.max() - series.min()
    if spread == 0:
        return series.where(series.isna(), 50.0)
    return (series - series.min()) / spread * 100


def main():
    (BASE_DIR / "data/processed").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "models").mkdir(exist_ok=True)
    # 정제 데이터 불러오기

    df = pd.read_csv(
        BASE_DIR / 'data/processed/clean_mart.csv'
    )

    print("=== 데이터 불러오기 완료 ===")
    print(df.shape)

    # 점포당 월매출 점수
    # 높을수록 좋음

    df["매출점수"] = minmax_score(
        df["점포당_월매출"]
    )

    # 유동인구 점수
    # 높을수록 좋음

    df["유동인구점수"] = minmax_score(
        df["총_유동인구_수"]
    )

    # 2030 점수
    # 높을수록 좋음

    df["2030점수"] = minmax_score(
        df["2030비율"]
    )

    # 폐업 안정성 점수
    # 폐업비율이 낮을수록 점수 높음

    df["폐업안정점수"] = (
        100
        - minmax_score(df["폐업비율"])
    )

    # 창업 추천 점수 계산

    df["창업추천점수"] = (
        df["매출점수"] * 0.40
        + df["유동인구점수"] * 0.25
        + df["2030점수"] * 0.20
        + df["폐업안정점수"] * 0.15
    )

    # 추천 순위 확인

    ranking = df.sort_values(
        "창업추천점수",
        ascending=False
    )

    print("\n=== 창업 추천 TOP 20 ===")

    print(
        ranking[
            [
                "상권_코드_명",
                "서비스_업종_코드_명",
                "점포당_월매출",
                "총_유동인구_수",
                "2030비율",
                "폐업비율",
                "창업추천점수"
            ]
        ].head(20)
    )

    # 점수 포함 데이터 저장

    df.to_csv(
        BASE_DIR / 'data/processed/scored_mart.csv',
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== 저장 완료 ===")
    print("data/processed/scored_mart.csv")


if __name__ == "__main__":
    main()
