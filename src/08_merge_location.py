import pandas as pd


# ==========================================
# 1. 데이터 불러오기
# ==========================================

area = pd.read_csv(
    "data/processed/area_location.csv"
)

startup = pd.read_csv(
    "data/processed/startup_ranking.csv"
)

hotplace = pd.read_csv(
    "data/processed/hotplace_score.csv"
)


print("=== 원본 데이터 크기 ===")
print("area:", area.shape)
print("startup:", startup.shape)
print("hotplace:", hotplace.shape)


# ==========================================
# 2. 위치 데이터에서 필요한 열만 선택
# ==========================================

location = area[
    [
        "상권_코드",
        "위도",
        "경도",
        "자치구_코드_명",
        "행정동_코드_명"
    ]
].copy()


# 상권코드 중복 제거
location = location.drop_duplicates(
    subset=["상권_코드"]
)


print("\n=== 위치 데이터 ===")
print(location.head())

print("\n위치 데이터 개수:")
print(len(location))


# ==========================================
# 3. 창업 추천 데이터 + 위치 병합
# ==========================================

startup_map = startup.merge(
    location,
    on="상권_코드",
    how="left"
)


print("\n=== 창업 추천 병합 결과 ===")
print(startup_map.shape)


print("\n창업 추천 위치 없는 행:")
print(
    startup_map["위도"].isna().sum()
)


# ==========================================
# 4. 핫플 데이터 + 위치 병합
# ==========================================

hotplace_map = hotplace.merge(
    location,
    on="상권_코드",
    how="left"
)


print("\n=== 핫플 병합 결과 ===")
print(hotplace_map.shape)


print("\n핫플 위치 없는 행:")
print(
    hotplace_map["위도"].isna().sum()
)


# ==========================================
# 5. 결과 확인
# ==========================================

print("\n=== 창업 추천 좌표 예시 ===")

print(
    startup_map[
        [
            "상권_코드_명",
            "서비스_업종_코드_명",
            "위도",
            "경도"
        ]
    ].head(10)
)


print("\n=== 핫플 좌표 예시 ===")

print(
    hotplace_map[
        [
            "상권_코드_명",
            "위도",
            "경도",
            "핫플점수"
        ]
    ].head(10)
)


# ==========================================
# 6. 저장
# ==========================================

startup_map.to_csv(
    "data/processed/startup_ranking_map.csv",
    index=False,
    encoding="utf-8-sig"
)

hotplace_map.to_csv(
    "data/processed/hotplace_score_map.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\n==============================")
print("저장 완료")
print("==============================")

print(
    "data/processed/startup_ranking_map.csv"
)

print(
    "data/processed/hotplace_score_map.csv"
)