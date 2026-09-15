import streamlit as st
import pandas as pd
import pydeck as pdk
import joblib
import base64

from pathlib import Path


# ----------------------------
# 1. 페이지 설정
# ----------------------------

st.set_page_config(
    page_title="서울 상권 AI 추천",
    page_icon="🏙️",
    layout="wide"
)




# ----------------------------
# 2. 데이터 불러오기
# ----------------------------

startup = pd.read_csv(
    "data/processed/startup_ranking_map.csv"
)

hotplace = pd.read_csv(
    "data/processed/hotplace_score_map.csv"
)
ai_model = joblib.load(
    "models/final_sales_model.pkl"
)

# ----------------------------
# 3. 제목
# ----------------------------

st.title("🏙️ 서울 상권 AI 추천")

st.write(
    "서울 상권 데이터를 활용하여 "
    "창업 추천과 핫플 추천을 확인할 수 있습니다."
)


# ----------------------------
# 4. 메뉴 선택
# ----------------------------

mode = st.radio(
    "원하는 서비스를 선택하세요.",
    [
        "👔 사장님 모드",
        "🔥 소비자 모드"
    ],
    horizontal=True
)
# =====================================
# 🎨 모드별 배경
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

if mode == "👔 사장님 모드":
    background_path = BASE_DIR / "assets" / "01.jpg"

else:
    background_path = BASE_DIR / "assets" / "02.png"

with open(background_path, "rb") as image_file:
    encoded_image = base64.b64encode(
        image_file.read()
    ).decode()

st.markdown(
    f"""
    <style>

    /* =====================================
       전체 배경
       ===================================== */

    .stApp {{
        background-image:
            linear-gradient(
                rgba(255,255,255,0.72),
                rgba(255,255,255,0.72)
            ),
            url("data:image/jpg;base64,{encoded_image}");

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}


    /* =====================================
       🔥 핫플 상세 카드
       ===================================== */

        .hotplace-card {{
        background: rgba(255, 255, 255, 0.68);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);

        border-radius: 18px;
        padding: 22px;
        margin-top: 10px;
        margin-bottom: 15px;

        border: 1px solid rgba(255, 255, 255, 0.55);

        box-shadow:
            0 6px 18px rgba(0, 0, 0, 0.08);
    }}

    .hotplace-number {{
        font-size: 30px;
        font-weight: 800;
        color: #17384c;
        margin-top: 8px;
    }}

    .hotplace-label {{
        font-size: 15px;
        color: #334e5c;
        font-weight: 700;
    }}

    .hotplace-info {{
        background: rgba(230, 242, 244, 0.68);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);

        border-radius: 16px;
        padding: 20px;

        border-left: 5px solid #2F6F73;
        border-top: 1px solid rgba(255,255,255,0.5);
        border-right: 1px solid rgba(255,255,255,0.5);
        border-bottom: 1px solid rgba(255,255,255,0.5);
    }}


    </style>
    """,
    unsafe_allow_html=True
)

  


# ==================================================
# 사장님 모드
# ==================================================
if mode == "👔 사장님 모드":

    st.header("👔 창업 추천")

    st.write(
        "업종을 선택하면 "
        "추천 상권 TOP 10을 보여드립니다."
    )


    # ----------------------------
    # 업종 목록
    # ----------------------------

    industry_list = sorted(
        startup["서비스_업종_코드_명"]
        .dropna()
        .unique()
    )


    # ----------------------------
    # 업종 선택
    # ----------------------------

    selected_industry = st.selectbox(
        "창업하고 싶은 업종을 선택하세요.",
        industry_list
    )
         # =====================================
    # 💬 상권 AI 질문
    # =====================================

    st.divider()

    st.subheader("💬 상권 AI에게 질문하기")

    st.caption(
        f"현재 선택 업종: {selected_industry}"
    )

    user_question = st.text_input(
        "궁금한 내용을 입력하세요.",
        placeholder=f"예: {selected_industry} 창업하기 좋은 상권은 어디야?"
    )

    if st.button(
        "🤖 AI에게 질문",
        key="seller_ai_question"
    ):

        if user_question:
            st.write("질문:")
            st.write(user_question)

        else:
            st.warning("질문을 입력해주세요.")

    # =====================================
    # 추천 상권 데이터 만들기
    # =====================================

    result = startup[
        startup["서비스_업종_코드_명"]
        == selected_industry
    ].copy()



    # ----------------------------
    # 추천점수 기준 정렬
    # ----------------------------

    result = result.sort_values(
        "평균_창업추천점수",
        ascending=False
    )


    # ----------------------------
    # TOP 10
    # ----------------------------

    result = result.head(10)

    result = result.reset_index(drop=True)

    result["순위"] = result.index + 1


    # ----------------------------
    # AI 예상 월매출 계산
    # ----------------------------

    ai_input = result[
        [
            "평균_총_유동인구",
            "평균_2030비율",
            "평균_점포_수",
            "평균_폐업비율",
            "서비스_업종_코드_명",
            "상권_코드_명"
        ]
    ].copy()


    # 모델이 학습한 컬럼 이름으로 변경
    ai_input = ai_input.rename(
        columns={
            "평균_총_유동인구": "총_유동인구_수",
            "평균_2030비율": "2030비율",
            "평균_점포_수": "점포_수",
            "평균_폐업비율": "폐업비율"
        }
    )


    # AI 예상 월매출 계산
    result["AI_예상_월매출"] = ai_model.predict(
        ai_input
    )


    # ----------------------------
    # TOP 10 제목
    # ----------------------------

    st.subheader(
        f"🏆 {selected_industry} 추천 상권 TOP 10"
    )


    # ----------------------------
    # 보기 좋은 열 만들기
    # ----------------------------

    display_result = result[
        [
            "순위",
            "상권_코드_명",
            "평균_점포당_월매출",
            "평균_총_유동인구",
            "평균_2030비율",
            "평균_폐업비율",
            "평균_창업추천점수",
            "AI_예상_월매출"
        ]
    ].copy()


   

    # ----------------------------
    # 표 출력
    # ----------------------------

    st.dataframe(
        display_result,
        use_container_width=True,
        hide_index=True,

        column_config={

            "순위": st.column_config.NumberColumn(
                "순위",
                format="%d"
            ),

            "평균 점포당 월매출": st.column_config.NumberColumn(
                "평균 점포당 월매출",
                format="%,.0f원"
            ),

            "평균 유동인구": st.column_config.NumberColumn(
                "평균 유동인구",
                format="%,.0f명"
            ),

            "2030 비율": st.column_config.NumberColumn(
                "2030 비율",
                format="%.1f%%"
            ),

            "폐업 비율": st.column_config.NumberColumn(
                "폐업 비율",
                format="%.1f%%"
            ),

            "추천 점수": st.column_config.NumberColumn(
                "추천 점수",
                format="%.1f점"
            ),
            "AI 예상 월매출": st.column_config.NumberColumn(
                "🤖 AI 예상 월매출",
                format="%,.0f원"
            )
        }
    )
      # ----------------------------
        # 추천 상권 지도
        # ----------------------------
    
    st.subheader("🗺️ 추천 상권 TOP 10 위치")
    
    seller_map = result[
            [
                "순위",
                "상권_코드_명",
                "위도",
                "경도",
                "평균_창업추천점수",
                "평균_점포당_월매출",
                "AI_예상_월매출",
                "평균_총_유동인구",
                "평균_2030비율"
            ]
        ].copy()
    
    
        # ----------------------------
        # 위도/경도 없는 데이터 제거
        # ----------------------------
    
    seller_map = seller_map.dropna(
            subset=["위도", "경도"]
        )
    
    
        # ----------------------------
        # 순위에 따라 점 크기 설정
        # ----------------------------
    
        # 기본: 6~10위
    seller_map["점크기"] = 500
    
        # 2~5위
    seller_map.loc[
            (seller_map["순위"] >= 2)
            & (seller_map["순위"] <= 5),
            "점크기"
        ] = 800
    
        # 1위
    seller_map.loc[
            seller_map["순위"] == 1,
            "점크기"
        ] = 1200
    
    
        # ----------------------------
        # 순위에 따라 색상 + 투명도
        # ----------------------------
    
    seller_map["색상"] = seller_map["순위"].apply(
            lambda rank:
            [30, 120, 255, 210] if rank == 1
            else [80, 150, 255, 150] if rank <= 5
            else [130, 190, 255, 90]
        )
    seller_map["추천점수표시"] = (
            seller_map["평균_창업추천점수"]
            .round(1)
            .astype(str)
            + "점"
        )
    seller_map["실제매출표시"] = (
        seller_map["평균_점포당_월매출"]
        .round(0)
        .map("{:,.0f}원".format)
    )

    seller_map["AI매출표시"] = (
        seller_map["AI_예상_월매출"]
        .round(0)
        .map("{:,.0f}원".format)
    )
    
    seller_map["월매출표시"] = (
            seller_map["평균_점포당_월매출"]
            .round(0)
            .map("{:,.0f}원".format)
        )
    
    seller_map["유동인구표시"] = (
            seller_map["평균_총_유동인구"]
            .round(0)
            .map("{:,.0f}명".format)
        )
    
    seller_map["2030표시"] = (
            seller_map["평균_2030비율"]
            .round(1)
            .astype(str)
            + "%"
        )
    
    
        # ----------------------------
        # 지도 점 만들기
        # ----------------------------
    
    seller_layer = pdk.Layer(
            "ScatterplotLayer",
            data=seller_map,
            get_position="[경도, 위도]",
            get_radius="점크기",
            get_fill_color="색상",
            pickable=True,
            stroked=True,
            get_line_color=[255, 255, 255, 180],
            line_width_min_pixels=1
        )
    
    
        # ----------------------------
        # 지도 시작 위치
        # ----------------------------
    
    seller_view = pdk.ViewState(
            latitude=seller_map["위도"].mean(),
            longitude=seller_map["경도"].mean(),
            zoom=11
        )
    
    
        # ----------------------------
        # 마우스를 올렸을 때 설명
        # ----------------------------
    
    seller_tooltip = {
            "html": """
            <b>🏆 순위:</b> {순위}위<br/>
            <b>상권:</b> {상권_코드_명}<br/>
            <b>추천 점수:</b> {추천점수표시}<br/>
            <b>실제 평균 월매출:</b> {실제매출표시}<br/>
            <b>🤖 AI 예상 월매출:</b> {AI매출표시}
            """
        }
    
        # ----------------------------
        # 지도 출력
        # ----------------------------
    
    
    st.pydeck_chart(
            pdk.Deck(
                layers=[seller_layer],
                initial_view_state=seller_view,
                tooltip=seller_tooltip
            )
        )


    # ----------------------------
    # TOP 10 추천 점수 그래프
    # ----------------------------

    st.subheader("📊 추천 상권 TOP 10 점수 비교")

    chart_data = result[
        [
            "상권_코드_명",
            "평균_창업추천점수"
        ]
    ].copy()

    chart_data = chart_data.set_index(
        "상권_코드_명"
    )

    st.bar_chart(
        chart_data,
        color="#2F6F73"
    )
      

      

    # ----------------------------
    # 1위 추천
    # ----------------------------

    if len(result) > 0:

        top = result.iloc[0]

        st.success(
            f"🥇 가장 추천하는 상권은 "
            f"'{top['상권_코드_명']}' 입니다."
        )
        st.subheader("🤖 AI 매출 예측")

        col1, col2 = st.columns(2)

        col1.metric(
            "AI 예상 월매출",
            f"{top['AI_예상_월매출']:,.0f}원"
        )

        col2.metric(
            "실제 평균 월매출",
            f"{top['평균_점포당_월매출']:,.0f}원"
        )
        sales_gap = (
            top["AI_예상_월매출"]
            - top["평균_점포당_월매출"]
        )
        error_rate = (
            abs(sales_gap)
            / top["평균_점포당_월매출"]
            * 100
        )
        st.metric(
            "AI 예측 오차율",
            f"{error_rate:.1f}%"
        )
        if sales_gap >= 0:

            st.info(
                f"🤖 AI 예상 매출이 실제 평균보다 "
                f"{sales_gap:,.0f}원 높습니다."
            )

        else:

            st.info(
                f"🤖 AI 예상 매출이 실제 평균보다 "
                f"{abs(sales_gap):,.0f}원 낮습니다."
            )


        # ----------------------------
        # 추천 이유 설명
        # ----------------------------

        st.subheader("💡 왜 이 상권을 추천하나요?")

        st.write(
            f"**{top['상권_코드_명']}**에서 "
            f"**{selected_industry}** 업종을 분석한 결과입니다."
        )


        col1, col2 = st.columns(2)


        with col1:

            st.info(
                f"""
💰 **매출 경쟁력**

평균 점포당 월매출은  
**{top['평균_점포당_월매출']:,.0f}원**입니다.
"""
            )

            st.info(
                f"""
👥 **유동인구**

평균 유동인구는  
**{top['평균_총_유동인구']:,.0f}명**입니다.
"""
            )


        with col2:

            st.info(
                f"""
🧑‍🤝‍🧑 **2030 고객 비율**

20~30대 유동인구 비율은  
**{top['평균_2030비율']:.1f}%**입니다.
"""
            )

            st.info(
                f"""
🏪 **상권 안정성**

평균 폐업 비율은  
**{top['평균_폐업비율']:.1f}%**입니다.
"""
            )


        
        # =====================================
            # 🤖 실제 매출 vs AI 예상 매출 비교
            # =====================================
        
        st.subheader("🤖 실제 평균 월매출 vs AI 예상 월매출")
        
        sales_compare = result[
                [
                    "상권_코드_명",
                    "평균_점포당_월매출",
                    "AI_예상_월매출"
                ]
            ].copy()
        
        sales_compare = sales_compare.set_index(
                "상권_코드_명"
            )
        
        st.bar_chart(
                sales_compare
            )
        
    
            
              


# ==================================================
# 소비자 모드
# ==================================================

else:

    st.header("🔥 서울 핫플 추천")

    st.write(
        "유동인구, 2030 비율, "
        "주말·야간 유동인구를 기준으로 "
        "핫플 상권을 추천합니다."
    )
    st.divider()

    st.subheader("🔥 핫플 AI에게 질문하기")

    consumer_question = st.text_input(
        "궁금한 내용을 입력하세요.",
        placeholder="예: 2030이 많이 가는 핫플은 어디야?",
        key="consumer_ai_question_input"
    )
    latest_year = hotplace["기준_년_코드"].max()

    latest_quarter = hotplace[
        hotplace["기준_년_코드"] == latest_year
    ]["기준_분기_코드"].max()

    latest = hotplace[
        (hotplace["기준_년_코드"] == latest_year)
        &
        (hotplace["기준_분기_코드"] == latest_quarter)
    ].copy()

    if st.button(
        "🤖 AI에게 질문",
        key="consumer_ai_question_button"
    ):

        if consumer_question:

            st.subheader("🤖 AI 분석 결과")

            question = consumer_question.replace(" ", "")

            # =====================================
            # 1. 주말 유동인구 질문
            # =====================================

            if (
                "주말" in question
                and
                ("유동" in question or "사람" in question)
            ):

                weekend_result = latest.sort_values(
                    "주말_유동인구_수",
                    ascending=False
                ).head(5)

                st.success(
                    "📅 주말 유동인구가 많은 상권 TOP 5입니다."
                )

                for i, (_, row) in enumerate(
                    weekend_result.iterrows(),
                    start=1
                ):

                    st.write(
                        f"**{i}위. {row['상권_코드_명']}** "
                        f"— {row['주말_유동인구_수']:,.0f}명"
                    )


            # =====================================
            # 2. 야간 유동인구 질문
            # =====================================

            elif (
                "야간" in question
                or "밤" in question
                or "저녁" in question
            ):

                night_result = latest.sort_values(
                    "야간_유동인구_수",
                    ascending=False
                ).head(5)

                st.success(
                    "🌙 야간 유동인구가 많은 상권 TOP 5입니다."
                )

                for i, (_, row) in enumerate(
                    night_result.iterrows(),
                    start=1
                ):

                    st.write(
                        f"**{i}위. {row['상권_코드_명']}** "
                        f"— {row['야간_유동인구_수']:,.0f}명"
                    )


            # =====================================
            # 3. 2030 질문
            # =====================================

            elif (
                "2030" in question
                or "20대" in question
                or "30대" in question
                or "젊은" in question
            ):

                young_result = latest.sort_values(
                    "2030비율",
                    ascending=False
                ).head(5)

                st.success(
                    "👨‍👩‍👧 2030 비율이 높은 상권 TOP 5입니다."
                )

                for i, (_, row) in enumerate(
                    young_result.iterrows(),
                    start=1
                ):

                    st.write(
                        f"**{i}위. {row['상권_코드_명']}** "
                        f"— 2030 비율 "
                        f"{row['2030비율']:.1f}%"
                    )


            # =====================================
            # 4. 특정 상권 비교
            # =====================================

            else:

                found_areas = []

                for area_name in latest[
                    "상권_코드_명"
                ].dropna().unique():

                    # 정확한 상권명
                    if area_name.replace(" ", "") in question:
                        found_areas.append(area_name)

                    # '역'을 제외한 이름도 검색
                    elif (
                        area_name
                        .replace("역", "")
                        .replace(" ", "")
                        in question
                    ):
                        found_areas.append(area_name)


                if len(found_areas) > 0:

                    compare_result = latest[
                        latest["상권_코드_명"].isin(
                            found_areas
                        )
                    ].copy()

                    compare_result = compare_result.sort_values(
                        "핫플점수",
                        ascending=False
                    )

                    for _, row in compare_result.iterrows():

                        st.write(
                            f"🔥 **{row['상권_코드_명']}**"
                        )

                        st.write(
                            f"핫플 점수: "
                            f"{row['핫플점수']:.1f}점"
                        )

                        st.write(
                            f"총 유동인구: "
                            f"{row['총_유동인구_수']:,.0f}명"
                        )

                        st.write(
                            f"2030 비율: "
                            f"{row['2030비율']:.1f}%"
                        )

                        st.write("---")


                    if len(compare_result) >= 2:

                        best = compare_result.iloc[0]

                        st.success(
                            f"🏆 현재 데이터 기준으로 "
                            f"**{best['상권_코드_명']}**의 "
                            f"핫플 점수가 더 높습니다."
                        )


                # =====================================
                # 5. 일반 핫플 질문
                # =====================================

                elif (
                    "핫플" in question
                    or "인기" in question
                    or "추천" in question
                    or "좋은곳" in question
                ):

                    hot_result = latest.sort_values(
                        "핫플점수",
                        ascending=False
                    ).head(5)

                    st.success(
                        "🔥 현재 데이터 기준 핫플 TOP 5입니다."
                    )

                    for i, (_, row) in enumerate(
                        hot_result.iterrows(),
                        start=1
                    ):

                        st.write(
                            f"**{i}위. {row['상권_코드_명']}** "
                            f"— 핫플점수 "
                            f"{row['핫플점수']:.1f}점"
                        )


                else:

                    st.info(
                        "질문을 이해하지 못했습니다. "
                        "예: '주말 유동인구 많은 곳', "
                        "'밤에 사람 많은 곳', "
                        "'2030이 많은 곳', "
                        "'홍대와 신촌 중 어디가 더 핫해?'"
                    )

        else:

            st.warning("질문을 입력해주세요.")




    # ----------------------------
    # 가장 최근 년/분기
    # ----------------------------

    latest_year = hotplace[
        "기준_년_코드"
    ].max()


    latest_quarter = hotplace[
        hotplace["기준_년_코드"]
        == latest_year
    ]["기준_분기_코드"].max()


    latest = hotplace[
        (
            hotplace["기준_년_코드"]
            == latest_year
        )
        &
        (
            hotplace["기준_분기_코드"]
            == latest_quarter
        )
    ].copy()


    # ----------------------------
    # 핫플 순위
    # ----------------------------

    latest = latest.sort_values(
        "핫플점수",
        ascending=False
    )

    latest = latest.head(20)

    latest = latest.reset_index(drop=True)

    latest["순위"] = latest.index + 1


    st.subheader(
        f"🔥 {latest_year}년 "
        f"{latest_quarter}분기 핫플 TOP 20"
    )


    # ----------------------------
    # 출력 데이터
    # ----------------------------

    display_hot = latest[
        [
            "순위",
            "상권_코드_명",
            "총_유동인구_수",
            "2030비율",
            "주말_유동인구_수",
            "야간_유동인구_수",
            "핫플점수"
        ]
    ].copy()


    display_hot.columns = [
        "순위",
        "상권",
        "총 유동인구",
        "2030 비율",
        "주말 유동인구",
        "야간 유동인구",
        "핫플 점수"
    ]


    # ----------------------------
    # 표 출력
    # ----------------------------

    st.dataframe(
        display_hot,
        use_container_width=True,
        hide_index=True,
        column_config={

            "순위": st.column_config.NumberColumn(
                "순위",
                format="%d"
            ),

            "총 유동인구": st.column_config.NumberColumn(
                "총 유동인구",
                format="%,.0f명"
            ),

            "2030 비율": st.column_config.NumberColumn(
                "2030 비율",
                format="%.1f%%"
            ),

            "주말 유동인구": st.column_config.NumberColumn(
                "주말 유동인구",
                format="%,.0f명"
            ),

            "야간 유동인구": st.column_config.NumberColumn(
                "야간 유동인구",
                format="%,.0f명"
            ),

            "핫플 점수": st.column_config.NumberColumn(
                "핫플 점수",
                format="%.1f점"
            )
        }
    )
     
   
    
    # =====================================
    # 🗺️ 서울 핫플 TOP 20 위치
    # =====================================

    st.subheader("🗺️ 서울 핫플 TOP 20 위치")

    # 지도에 사용할 데이터
    consumer_map = latest[
        [
            "순위",
            "상권_코드_명",
            "위도",
            "경도",
            "핫플점수",
            "총_유동인구_수",
            "2030비율"
        ]
    ].copy()

    # 위도/경도 없는 데이터 제거
    consumer_map = consumer_map.dropna(
        subset=["위도", "경도"]
    )

    # 점 크기
    consumer_map["점크기"] = 500

    consumer_map.loc[
        (consumer_map["순위"] >= 2)
        & (consumer_map["순위"] <= 5),
        "점크기"
    ] = 800

    consumer_map.loc[
        consumer_map["순위"] == 1,
        "점크기"
    ] = 1200

    # 점 색상
    consumer_map["색상"] = consumer_map["순위"].apply(
        lambda rank:
        [255, 60, 60, 210]
        if rank == 1
        else [255, 120, 80, 150]
        if rank <= 5
        else [255, 180, 100, 90]
    )

    # 툴팁용 문자열
    consumer_map["핫플점수표시"] = (
        consumer_map["핫플점수"]
        .round(1)
        .astype(str)
        + "점"
    )

    consumer_map["유동인구표시"] = (
        consumer_map["총_유동인구_수"]
        .round(0)
        .map("{:,.0f}명".format)
    )

    consumer_map["2030표시"] = (
        consumer_map["2030비율"]
        .round(1)
        .astype(str)
        + "%"
    )

    # 지도 점
    consumer_layer = pdk.Layer(
        "ScatterplotLayer",
        data=consumer_map,
        get_position="[경도, 위도]",
        get_radius="점크기",
        get_fill_color="색상",
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255, 180],
        line_width_min_pixels=1
    )

    # 지도 시작 위치
    consumer_view = pdk.ViewState(
        latitude=consumer_map["위도"].mean(),
        longitude=consumer_map["경도"].mean(),
        zoom=11
    )

    # 마우스 툴팁
    consumer_tooltip = {
        "html": """
        <b>🔥 순위:</b> {순위}위<br/>
        <b>상권:</b> {상권_코드_명}<br/>
        <b>핫플 점수:</b> {핫플점수표시}<br/>
        <b>총 유동인구:</b> {유동인구표시}<br/>
        <b>2030 비율:</b> {2030표시}
        """
    }

    # 지도 출력
    st.pydeck_chart(
        pdk.Deck(
            layers=[consumer_layer],
            initial_view_state=consumer_view,
            tooltip=consumer_tooltip
        ),
        use_container_width=True
    )


    # =====================================
    # 🏪 지역별 인기 업종 TOP 10
    # =====================================

    st.divider()

    st.subheader("🏪 지역별 인기 업종 TOP 10")

    area_list = (
        latest["상권_코드_명"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_area = st.selectbox(
        "지역을 선택하세요.",
        area_list,
        key="consumer_area_select"
    )

    area_shop = startup[
        startup["상권_코드_명"] == selected_area
    ].copy()

    area_shop = area_shop.sort_values(
        "평균_점포당_월매출",
        ascending=False
    ).head(10)

    area_shop = area_shop.reset_index(drop=True)

    area_shop["순위"] = area_shop.index + 1

    shop_display = area_shop[
        [
            "순위",
            "서비스_업종_코드_명",
            "평균_점포당_월매출",
            "평균_총_유동인구",
            "평균_2030비율"
        ]
    ].copy()

    shop_display.columns = [
        "순위",
        "업종",
        "평균 월매출",
        "평균 유동인구",
        "2030 비율"
    ]

    st.dataframe(
        shop_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "평균 월매출": st.column_config.NumberColumn(
                "평균 월매출",
                format="%,.0f원"
            ),
            "평균 유동인구": st.column_config.NumberColumn(
                "평균 유동인구",
                format="%,.0f명"
            ),
            "2030 비율": st.column_config.NumberColumn(
                "2030 비율",
                format="%.1f%%"
            )
        }
    )


    # =====================================
    # 📊 서울 핫플 TOP 20 점수 비교
    # =====================================

    st.divider()

    st.subheader("📊 서울 핫플 TOP 20 점수 비교")

    hot_chart = latest[
        [
            "상권_코드_명",
            "핫플점수"
        ]
    ].copy()

    hot_chart = hot_chart.set_index(
        "상권_코드_명"
    )

    st.bar_chart(
        hot_chart,
        color="#D9795F"
    )

    

   
    
      # =====================================
    # 🔥 1위 핫플 상세
    # =====================================

    if len(latest) > 0:

        top = latest.iloc[0]

        st.divider()

        st.subheader("🔥 top 10핫플 상세")

        # -----------------------------
        # 1위 상권 소개
        # -----------------------------

        top_card = f"""
<div class="hotplace-card">
<div style="font-size:14px; font-weight:700; color:#B95F47;">
🔥 2025년 4분기 핫플 1위
</div>




<div style="font-size:15px; color:#455a64; margin-top:8px;">
유동인구 · 2030 비율 · 주말 · 야간 방문 특성을 종합한 결과입니다.
</div>
</div>
"""
       # =====================================
    # 🔥 TOP 10 핫플 상세
    # =====================================

    if len(latest) > 0:

        detail_top10 = latest.head(10).copy()

        detail_top10["선택메뉴"] = (
            detail_top10["순위"].astype(str)
            + "위 - "
            + detail_top10["상권_코드_명"]
        )

        # ---------------------------------
        # TOP 10 역 선택
        # ---------------------------------

        selected_station = st.selectbox(
            "핫플 역 선택",
            detail_top10["선택메뉴"].tolist(),
            key="hotplace_station_select",
            label_visibility="collapsed"
        )

        # 선택한 역 데이터
        top = detail_top10[
            detail_top10["선택메뉴"] == selected_station
        ].iloc[0]

        top_rank = int(top["순위"])


        # ---------------------------------
        # 선택한 핫플 소개
        # ---------------------------------

        st.markdown(
            f"""
<div class="hotplace-card">

<div style="
    font-size:14px;
    font-weight:700;
    color:#B95F47;
">
🔥 2025년 4분기 핫플 {top_rank}위
</div>

<div style="
    font-size:26px;
    font-weight:800;
    color:#17384c;
    margin-top:8px;
">
{top['상권_코드_명']}
</div>

<div style="
    font-size:15px;
    color:#455a64;
    margin-top:8px;
">
유동인구 · 2030 비율 · 주말 · 야간 방문 특성을 종합한 결과입니다.
</div>

</div>
""",
            unsafe_allow_html=True
        )

        # =====================================
        # 핵심 지표
        # =====================================

        col1, col2, col3 = st.columns(3)

        with col1:

            card1 = f"""
<div class="hotplace-card" style="text-align:center;">
<div class="hotplace-label">
🔥 핫플 점수
</div>

<div class="hotplace-number">
{top['핫플점수']:.1f}점
</div>
</div>
"""

            st.markdown(
                card1,
                unsafe_allow_html=True
            )


        with col2:

            card2 = f"""
<div class="hotplace-card" style="text-align:center;">
<div class="hotplace-label">
👥 총 유동인구
</div>

<div class="hotplace-number">
{top['총_유동인구_수']:,.0f}명
</div>
</div>
"""

            st.markdown(
                card2,
                unsafe_allow_html=True
            )


        with col3:

            card3 = f"""
<div class="hotplace-card" style="text-align:center;">
<div class="hotplace-label">
🧑‍🤝‍🧑 2030 비율
</div>

<div class="hotplace-number">
{top['2030비율']:.1f}%
</div>
</div>
"""

            st.markdown(
                card3,
                unsafe_allow_html=True
            )


        # =====================================
        # 왜 여기가 핫플인가요?
        # =====================================

        st.subheader("💡 왜 여기가 핫플인가요?")

        col4, col5 = st.columns(2)

        with col4:

            weekend_card = f"""
<div class="hotplace-info">
<div style="font-size:17px; font-weight:700; color:#17384c;">
🎉 주말 유동인구
</div>

<div style="font-size:14px; margin-top:10px;">
토요일 + 일요일 방문자
</div>

<div style="font-size:27px; font-weight:800; color:#17384c; margin-top:6px;">
{top['주말_유동인구_수']:,.0f}명
</div>
</div>
"""

            st.markdown(
                weekend_card,
                unsafe_allow_html=True
            )


        with col5:

            night_card = f"""
<div class="hotplace-info">
<div style="font-size:17px; font-weight:700; color:#17384c;">
🌙 야간 유동인구
</div>

<div style="font-size:14px; margin-top:10px;">
17시 ~ 24시 방문자
</div>

<div style="font-size:27px; font-weight:800; color:#17384c; margin-top:6px;">
{top['야간_유동인구_수']:,.0f}명
</div>
</div>
"""

            st.markdown(
                night_card,
                unsafe_allow_html=True
            )