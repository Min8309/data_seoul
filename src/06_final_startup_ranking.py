import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def main():
    (BASE_DIR / "data/processed").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "models").mkdir(exist_ok=True)
    # 개선된 점수 데이터 불러오기

    df = pd.read_csv(
        BASE_DIR / 'data/processed/scored_mart_v2.csv'
    )

    print("=== 데이터 불러오기 완료 ===")
    print(df.shape)

    # 점포가 너무 적은 업종 제외

    df = df[
        df["점포_수"] >= 3
    ].copy()

    # 상권 + 업종 단위로 평균 계산

    startup = (
        df.groupby(
            [
                "상권_코드",
                "상권_코드_명",
                "서비스_업종_코드",
                "서비스_업종_코드_명"
            ],
            as_index=False
        )
        .agg(
            평균_점포_수=("점포_수", "mean"),
            평균_점포당_월매출=("점포당_월매출", "mean"),
            평균_총_유동인구=("총_유동인구_수", "mean"),
            평균_2030비율=("2030비율", "mean"),
            평균_폐업비율=("폐업비율", "mean"),
            평균_창업추천점수=("창업추천점수_v2", "mean")
        )
    )

    # 추천 순위 정렬

    startup = startup.sort_values(
        "평균_창업추천점수",
        ascending=False
    )

    # TOP 20 출력

    print("\n=== 최종 창업 추천 TOP 20 ===")

    print(
        startup[
            [
                "상권_코드_명",
                "서비스_업종_코드_명",
                "평균_점포_수",
                "평균_점포당_월매출",
                "평균_총_유동인구",
                "평균_2030비율",
                "평균_폐업비율",
                "평균_창업추천점수"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    # 순위 번호 만들기

    startup["추천순위"] = range(
        1,
        len(startup) + 1
    )

    # 저장

    startup.to_csv(
        BASE_DIR / 'data/processed/startup_ranking.csv',
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== 저장 완료 ===")
    print("data/processed/startup_ranking.csv")


if __name__ == "__main__":
    main()
