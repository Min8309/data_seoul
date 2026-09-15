import pandas as pd
from pyproj import Transformer


# ----------------------------
# 1. 상권 영역 데이터 불러오기
# ----------------------------

area = pd.read_csv(
    "data/raw/area.csv",
    encoding="cp949"
)

print("=== area.csv 불러오기 완료 ===")
print(area.shape)


# ----------------------------
# 2. 필요한 열만 선택
# ----------------------------

area = area[
    [
        "상권_코드",
        "상권_코드_명",
        "엑스좌표_값",
        "와이좌표_값",
        "자치구_코드_명",
        "행정동_코드_명"
    ]
].copy()


# ----------------------------
# 3. 좌표 변환기 만들기
# EPSG:5181 → EPSG:4326
# ----------------------------

transformer = Transformer.from_crs(
    "EPSG:5181",
    "EPSG:4326",
    always_xy=True
)


# ----------------------------
# 4. X/Y → 경도/위도 변환
# ----------------------------

longitude, latitude = transformer.transform(
    area["엑스좌표_값"].values,
    area["와이좌표_값"].values
)

area["경도"] = longitude
area["위도"] = latitude


# ----------------------------
# 5. 결과 확인
# ----------------------------

print("\n=== 좌표 변환 결과 ===")

print(
    area[
        [
            "상권_코드",
            "상권_코드_명",
            "위도",
            "경도"
        ]
    ].head(10)
)


# ----------------------------
# 6. 서울 범위에 들어오는지 확인
# ----------------------------

print("\n=== 위도 범위 ===")
print(area["위도"].min())
print(area["위도"].max())

print("\n=== 경도 범위 ===")
print(area["경도"].min())
print(area["경도"].max())


# ----------------------------
# 7. 저장
# ----------------------------

area.to_csv(
    "data/processed/area_location.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== 저장 완료 ===")
print("data/processed/area_location.csv")