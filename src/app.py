import streamlit as st
import pandas as pd
import pydeck as pdk
from html import escape
import altair as alt

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
LOGO_PATH = BASE_DIR / "assets/logo.png"

# 1. 페이지 설정

st.set_page_config(
    page_title="서울 어디가 | 서울 상권 탐색",
    page_icon=str(LOGO_PATH) if LOGO_PATH.is_file() else None,
    layout="wide"
)


# 2. 데이터 불러오기

@st.cache_data(show_spinner=False)
def load_startup_data():
    return pd.read_csv(BASE_DIR / "data/processed/startup_ranking_map.csv")


@st.cache_data(show_spinner=False)
def load_hotplace_data():
    return pd.read_csv(BASE_DIR / "data/processed/hotplace_score_map.csv")


@st.cache_resource
def load_model(model_version=None):
    import joblib

    return joblib.load(BASE_DIR / "models/final_sales_model.pkl")


@st.cache_data(show_spinner=False, max_entries=128)
def predict_sales(ai_input, model_version):
    """Reuse predictions until the input or saved model changes."""
    return load_model(model_version).predict(ai_input)


@st.cache_data
def load_styles(style_version=None):
    return (BASE_DIR / "assets/styles.css").read_text()


def show_rank_map(data, colors, tooltip):
    """Render either ranking with the same marker sizes and map settings."""
    data = data.dropna(subset=["위도", "경도"]).copy()
    data = data[data["위도"].between(-90, 90) & data["경도"].between(-180, 180)]
    if data.empty:
        st.info("표시할 위치 정보가 없습니다.")
        return
    data["상권_코드_명"] = data["상권_코드_명"].map(escape)
    data["점크기"] = data["순위"].map(lambda rank: 1200 if rank == 1 else 800 if rank <= 5 else 500)
    data["색상"] = data["순위"].map(lambda rank: colors[0] if rank == 1 else colors[1] if rank <= 5 else colors[2])
    st.pydeck_chart(
        pdk.Deck(
            layers=[pdk.Layer(
                "ScatterplotLayer", data=data, get_position="[경도, 위도]",
                get_radius="점크기", get_fill_color="색상", pickable=True,
                stroked=True, get_line_color=[255, 255, 255, 180], line_width_min_pixels=1,
            )],
            initial_view_state=pdk.ViewState(
                latitude=data["위도"].mean(), longitude=data["경도"].mean(), zoom=11,
            ),
            tooltip={"html": tooltip},
        ),
        width="stretch",
    )


def show_score_chart(data, score_column, title):
    base = alt.Chart(data).encode(
        x=alt.X("상권_코드_명:N", sort=data["상권_코드_명"].tolist(),
                title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f"{score_column}:Q", title=title, scale=alt.Scale(domain=[0, 100])),
    )
    chart = (
        base.mark_area(color=chart_color, opacity=0.15)
        + base.mark_line(color=chart_color, strokeWidth=3)
        + base.mark_circle(color=chart_color, size=110, stroke="white", strokeWidth=2).encode(
            tooltip=[alt.Tooltip("상권_코드_명:N", title="상권"),
                     alt.Tooltip(f"{score_column}:Q", title=title, format=".2f")],
        )
    ).properties(height=350).configure_view(strokeOpacity=0).configure_axis(
        gridColor="#E2E8F0", labelColor="#64748B", titleColor="#64748B",
    ).configure(background="#FFFFFF")
    st.altair_chart(chart, width="stretch")


def show_metric_card(label, value):
    st.markdown(
        f'<div class="hotplace-card" style="text-align:center;">'
        f'<div class="hotplace-label">{escape(label)}</div>'
        f'<div class="hotplace-number">{escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


# Dashboard header
header = st.container()
st.markdown("<style>" + load_styles((BASE_DIR / "assets/styles.css").stat().st_mtime_ns) + "</style>", unsafe_allow_html=True)

def navigate_to(consumer, tab):
    """Switch both the service and the selected analysis tab in one click."""
    st.session_state.mode_switch = consumer
    st.session_state["consumer_tabs" if consumer else "owner_tabs"] = tab


with st.sidebar:
    with st.container(key="sidebar_brand"):
        logo_col, title_col = st.columns([1, 3], gap="small", vertical_alignment="center")
        with logo_col:
            if LOGO_PATH.is_file():
                st.image(str(LOGO_PATH), width=64)
        with title_col:
            st.markdown("**서울 어디가**")
            st.caption("서울 상권 탐색 대시보드")
    st.divider()
    for consumer, heading, items in [
        (False, "👔 창업 추천", [
            ("개요 요약", "추천 요약 · AI 매출 예측"),
            ("상권 지도", "추천 상권 지도"),
            ("지표 비교", "창업 추천 점수 비교"),
            ("상세 데이터", "창업 추천 TOP 10 데이터"),
        ]),
        (True, "🔥 서울 핫플 추천", [
            ("개요 요약", "핫플 요약 · 상세 분석"),
            ("상권 지도", "서울 핫플 지도"),
            ("지역별 업종", "지역별 인기 업종"),
            ("상세 데이터", "핫플 TOP 20 데이터"),
        ]),
    ]:
        if consumer != st.session_state.get("mode_switch", False):
            continue
        st.markdown(f"#### {heading}")
        for tab, label in items:
            tab_key = "consumer_tabs" if consumer else "owner_tabs"
            active = (
                st.session_state.get("mode_switch", False) == consumer
                and st.session_state.get(tab_key, "개요 요약") == tab
            )
            st.button(
                label, key=f"nav_{tab_key}_{tab}",
                type="primary" if active else "tertiary",
                use_container_width=True,
                on_click=navigate_to, args=(consumer, tab),
            )
    st.divider()
    st.caption("현재 모드의 메뉴입니다. 클릭하면 해당 분석 탭이 열립니다.")

# 4. 메뉴 선택

st.markdown("원하는 서비스를 선택하세요.")

consumer_mode = st.session_state.get("mode_switch", False)
with st.container(key="mode_selector"):
    col_owner, col_consumer = st.columns(2, gap="small")
    with col_owner:
        st.button(
            "👔 사장님 모드  ✓" if not consumer_mode else "👔 사장님 모드  ↗",
            key="mode_owner_button", use_container_width=True,
            type="primary" if not consumer_mode else "secondary",
            on_click=navigate_to, args=(False, "개요 요약"),
            help="업종별 창업 추천과 AI 매출 예측 보기",
        )
    with col_consumer:
        st.button(
            "🔥 소비자 모드  ✓" if consumer_mode else "🔥 소비자 모드  ↗",
            key="mode_consumer_button", use_container_width=True,
            type="primary" if consumer_mode else "secondary",
            on_click=navigate_to, args=(True, "개요 요약"),
            help="서울 핫플과 지역별 인기 업종 보기",
        )

if consumer_mode:
    mode = "🔥 소비자 모드"
else:
    mode = "👔 사장님 모드"

# Render a mode marker so CSS switches without replacing native widgets.
chart_color = "#047857" if consumer_mode else "#2563EB"
theme_name = "consumer" if consumer_mode else "owner"
st.markdown(f'<span class="theme-marker theme-{theme_name}" aria-hidden="true"></span>', unsafe_allow_html=True)
with header:
    eyebrow = "CITY GUIDE / SEOUL" if consumer_mode else "BUSINESS INTELLIGENCE / SEOUL"
    title = "이번 주말, 서울 어디로 갈까?" if consumer_mode else "내 가게의 다음 기회, 데이터에서 찾다"
    description = (
        "젊은 에너지부터 밤의 활기까지, 취향에 맞는 서울의 핫플을 발견하세요."
        if consumer_mode else
        "업종별 상권 경쟁력과 AI 예상 매출을 비교하고, 창업 후보지를 좁혀보세요."
    )
    st.markdown(
        f'<div class="dashboard-hero"><div class="brand">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{description}</p>'
        f'<span class="mode-badge">{"도시 탐색 · 소비자 모드" if consumer_mode else "상권 분석 · 사장님 모드"}</span></div>',
        unsafe_allow_html=True,
    )
with st.sidebar:
    st.divider()
    st.caption("발견하는 서울 · 핫플·지역별 인기 업종" if consumer_mode else "분석하는 서울 · 업종별 추천·AI 매출 예측")

# 사장님 모드
if mode == "👔 사장님 모드":

    st.header("👔 창업 추천")

    st.write(
        "업종을 선택하면 "
        "추천 상권 TOP 10을 보여드립니다."
    )


    with st.spinner("창업 추천 데이터를 불러오는 중입니다…"):
        startup = load_startup_data()

    # 업종 목록

    industry_list = sorted(
        startup["서비스_업종_코드_명"]
        .dropna()
        .unique()
    )


    if not industry_list:
        st.info("창업 추천 데이터가 없습니다.")
        st.stop()

    # 업종 선택

    selected_industry = st.selectbox(
        "창업하고 싶은 업종을 선택하세요.",
        industry_list,
        key="industry_filter"
    )
    # 추천 상권 데이터 만들기

    result = startup[
        startup["서비스_업종_코드_명"]
        == selected_industry
    ].copy()


    # 추천점수 기준 정렬

    result = result.sort_values(
        "평균_창업추천점수",
        ascending=False
    )


    # TOP 10

    result = result.head(10)

    result = result.reset_index(drop=True)

    result["순위"] = result.index + 1


    # AI 예상 월매출 계산

    if result.empty:
        st.info("선택한 업종의 추천 데이터가 없습니다.")
        st.stop()

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
    model_version = (BASE_DIR / "models/final_sales_model.pkl").stat().st_mtime_ns
    with st.spinner("예상 월매출을 계산하는 중입니다…"):
        result["AI_예상_월매출"] = predict_sales(ai_input, model_version)


    st.caption("분석 범위 · 선택 업종의 추천 상권 TOP 10 · 매출은 점포당 월 기준")
    for col, label, value in zip(st.columns(4),
        ["추천 상권", "최고 추천 점수", "1위 AI 예상 월매출", "분석 업종"],
        [f"{len(result)}곳", f"{result['평균_창업추천점수'].max():.1f}점",
         f"{result.iloc[0]['AI_예상_월매출']:,.0f}원", selected_industry]):
        with col:
            show_metric_card(label, value)
    summary_tab, map_tab, compare_tab, data_tab = st.tabs(
        ["개요 요약", "상권 지도", "지표 비교", "상세 데이터"],
        key="owner_tabs", on_change="rerun",
    )

    with data_tab:
        # TOP 10 제목

        st.subheader(
            f"🏆 {selected_industry} 추천 상권 TOP 10"
        )


        # 보기 좋은 열 만들기

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


        # 표 출력

        st.dataframe(
            display_result,
            width="stretch",
            hide_index=True,

            column_config={

                "순위": st.column_config.NumberColumn(
                    "순위",
                    format="%d"
                ),

                "평균_점포당_월매출": st.column_config.NumberColumn(
                    "평균 점포당 월매출",
                    format="%,.0f원"
                ),

                "평균_총_유동인구": st.column_config.NumberColumn(
                    "평균 유동인구",
                    format="%,.0f명"
                ),

                "평균_2030비율": st.column_config.NumberColumn(
                    "2030 비율",
                    format="%.1f%%"
                ),

                "평균_폐업비율": st.column_config.NumberColumn(
                    "폐업 비율",
                    format="%.1f%%"
                ),

                "평균_창업추천점수": st.column_config.NumberColumn(
                    "추천 점수",
                    format="%.1f점"
                ),
                "AI_예상_월매출": st.column_config.NumberColumn(
                    "🤖 AI 예상 월매출",
                    format="%,.0f원"
                )
            }
        )
    with map_tab:
            # 추천 상권 지도

        st.subheader("🗺️ 추천 상권 TOP 10 위치")

        seller_map = result.copy()
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

        seller_tooltip = {
                "html": """
                <b>🏆 순위:</b> {순위}위<br/>
                <b>상권:</b> {상권_코드_명}<br/>
                <b>추천 점수:</b> {추천점수표시}<br/>
                <b>실제 평균 월매출:</b> {실제매출표시}<br/>
                <b>🤖 AI 예상 월매출:</b> {AI매출표시}
                """
            }

            # 지도 출력


        show_rank_map(seller_map, [[37, 99, 235, 220], [96, 165, 250, 180], [148, 163, 184, 140]], seller_tooltip["html"])

    with compare_tab:
        # TOP 10 추천 점수 그래프

        st.subheader("📈 추천 상권 TOP 10 점수 비교")

        show_score_chart(result, "평균_창업추천점수", "창업 추천 점수")

    with summary_tab:
        # 1위 추천

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
                if top["평균_점포당_월매출"] > 0
                else None
            )
            st.metric(
                "평균 매출 대비 예측 차이율",
                f"{error_rate:.1f}%" if error_rate is not None else "계산 불가"
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


            # 추천 이유 설명

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


                # 🤖 실제 매출 vs AI 예상 매출 비교

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


# 소비자 모드

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
    with st.spinner("핫플 데이터를 불러오는 중입니다…"):
        hotplace = load_hotplace_data()

    if hotplace.empty:
        st.info("핫플 추천 데이터가 없습니다.")
        st.stop()

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

        if consumer_question.strip():

            st.subheader("🤖 AI 분석 결과")

            question = "".join(consumer_question.split())

            # 1. 주말 유동인구 질문

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


            # 2. 야간 유동인구 질문

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


            # 3. 2030 질문

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


            # 4. 특정 상권 비교

            else:

                found_areas = []

                for area_name in latest[
                    "상권_코드_명"
                ].dropna().unique():

                    name = "".join(area_name.split())
                    short_name = name.removesuffix("역")
                    if name in question or (short_name and short_name in question):
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


                # 5. 일반 핫플 질문

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


    # 핫플 순위

    latest = latest.sort_values(
        "핫플점수",
        ascending=False
    )

    latest = latest.head(20)

    latest = latest.reset_index(drop=True)

    latest["순위"] = latest.index + 1


    st.caption(f"데이터 기준 · {latest_year}년 {latest_quarter}분기 · 핫플 TOP 20")
    for col, label, value in zip(st.columns(4),
        ["추천 상권", "최고 핫플 점수", "TOP 20 평균 2030 비율", "1위 상권"],
        [f"{len(latest)}곳", f"{latest['핫플점수'].max():.1f}점",
         f"{latest['2030비율'].mean():.1f}%", str(latest.iloc[0]['상권_코드_명'])]):
        with col:
            show_metric_card(label, value)
    summary_tab, map_tab, compare_tab, data_tab = st.tabs(
        ["개요 요약", "상권 지도", "지역별 업종", "상세 데이터"],
        key="consumer_tabs", on_change="rerun",
    )

    with data_tab:
        st.subheader(
            f"🔥 {latest_year}년 "
            f"{latest_quarter}분기 핫플 TOP 20"
        )


        # 출력 데이터

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


        # 표 출력

        st.dataframe(
            display_hot,
            width="stretch",
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


    with map_tab:
        # 🗺️ 서울 핫플 TOP 20 위치

        st.subheader("🗺️ 서울 핫플 TOP 20 위치")

        consumer_map = latest.copy()

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

        show_rank_map(consumer_map, [[234, 88, 12, 230], [5, 150, 105, 190], [110, 231, 183, 160]], consumer_tooltip["html"])

    with compare_tab:
        # 🏪 지역별 인기 업종 TOP 10

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

        startup = load_startup_data()
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
            width="stretch",
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


    with summary_tab:
        # 📊 서울 핫플 TOP 20 점수 비교

        st.divider()

        st.subheader("📈 서울 핫플 TOP 20 점수 비교")

        show_score_chart(latest, "핫플점수", "핫플 점수")

        # TOP 10 핫플 상세
        if len(latest) > 0:

            detail_top10 = latest.head(10).copy()

            detail_top10["선택메뉴"] = (
                detail_top10["순위"].astype(str)
                + "위 - "
                + detail_top10["상권_코드_명"]
            )

            # TOP 10 역 선택

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


            # 선택한 핫플 소개

            st.markdown(
                f"""
    <div class="hotplace-card">

    <div style="
        font-size:14px;
        font-weight:700;
        color:var(--accent);
    ">
    🔥 {latest_year}년 {latest_quarter}분기 핫플 {top_rank}위
    </div>

    <div style="
        font-size:26px;
        font-weight:800;
        color:#0F172A;
        margin-top:8px;
    ">
    {escape(top['상권_코드_명'])}
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

            # 핵심 지표

            col1, col2, col3 = st.columns(3)

            with col1:
                show_metric_card("🔥 핫플 점수", f"{top['핫플점수']:.1f}점")
            with col2:
                show_metric_card("👥 총 유동인구", f"{top['총_유동인구_수']:,.0f}명")
            with col3:
                show_metric_card("🧑‍🤝‍🧑 2030 비율", f"{top['2030비율']:.1f}%")

            # 왜 여기가 핫플인가요?

            st.subheader("💡 왜 여기가 핫플인가요?")

            col4, col5 = st.columns(2)

            with col4:

                weekend_card = f"""
    <div class="hotplace-info">
    <div style="font-size:17px; font-weight:700; color:#0F172A;">
    🎉 주말 유동인구
    </div>

    <div style="font-size:14px; margin-top:10px;">
    토요일 + 일요일 방문자
    </div>

    <div style="font-size:27px; font-weight:800; color:#0F172A; margin-top:6px;">
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
    <div style="font-size:17px; font-weight:700; color:#0F172A;">
    🌙 야간 유동인구
    </div>

    <div style="font-size:14px; margin-top:10px;">
    17시 ~ 24시 방문자
    </div>

    <div style="font-size:27px; font-weight:800; color:#0F172A; margin-top:6px;">
    {top['야간_유동인구_수']:,.0f}명
    </div>
    </div>
    """

                st.markdown(
                    night_card,
                    unsafe_allow_html=True
                )
