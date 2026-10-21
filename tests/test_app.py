import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class AppSmokeTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(ROOT / 'src/app.py'), default_timeout=60).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        return app

    def test_owner_predictions_and_column_configuration(self):
        app = self.app()
        for industry in [app.selectbox[0].options[0], app.selectbox[0].options[-1]]:
            app.selectbox[0].select(industry).run()
            self.assertFalse(app.exception)
            table = app.dataframe[0]
            self.assertEqual(len(table.value), 10)
            self.assertTrue(np.isfinite(table.value['AI_예상_월매출']).all())
            config = json.loads(table.proto.columns)
            self.assertEqual(config['AI_예상_월매출']['label'], '🤖 AI 예상 월매출')
        self.assertEqual(len(app.text_input), 0)

    def test_consumer_rankings_and_question_branches(self):
        app = self.app()
        app.toggle(key='mode_switch').set_value(True).run()
        self.assertFalse(app.exception)
        ranking = next(table.value for table in app.dataframe if '핫플 점수' in table.value.columns)
        self.assertEqual(len(ranking), 20)
        self.assertEqual(ranking['순위'].tolist(), list(range(1, 21)))
        for question in ['주말 유동인구 많은 곳', '밤에 사람 많은 곳', '2030이 많은 곳', '핫플 추천']:
            app.text_input(key='consumer_ai_question_input').set_value(question)
            app.button(key='consumer_ai_question_button').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.success)
        app.text_input(key='consumer_ai_question_input').set_value('')
        app.button(key='consumer_ai_question_button').click().run()
        self.assertTrue(app.warning)

    def test_dynamic_period_and_detail_selection(self):
        app = self.app()
        app.toggle(key='mode_switch').set_value(True).run()
        app.selectbox(key='hotplace_station_select').select_index(1).run()
        self.assertFalse(app.exception)
        data = pd.read_csv(ROOT / 'data/processed/hotplace_score_map.csv')
        year = data['기준_년_코드'].max()
        quarter = data.loc[data['기준_년_코드'] == year, '기준_분기_코드'].max()
        expected = f'{year}년 {quarter}분기 핫플 2위'
        self.assertTrue(any(expected in item.value for item in app.markdown))

    def test_whitespace_question_is_empty(self):
        app = self.app()
        app.toggle(key='mode_switch').set_value(True).run()
        app.text_input(key='consumer_ai_question_input').set_value(' \t ')
        app.button(key='consumer_ai_question_button').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.warning)

    def test_missing_coordinates_do_not_crash_either_mode(self):
        from unittest.mock import patch
        import streamlit as st
        original_read = pd.read_csv

        def without_coordinates(path, *args, **kwargs):
            data = original_read(path, *args, **kwargs)
            if Path(path).name in {'startup_ranking_map.csv', 'hotplace_score_map.csv'}:
                data[['위도', '경도']] = np.nan
            return data

        st.cache_data.clear()
        try:
            with patch('pandas.read_csv', side_effect=without_coordinates):
                app = self.app()
                self.assertTrue(any('위치 정보' in item.value for item in app.info))
                app.toggle(key='mode_switch').set_value(True).run()
                self.assertFalse(app.exception)
                self.assertTrue(any('위치 정보' in item.value for item in app.info))
        finally:
            st.cache_data.clear()

    def test_sidebar_navigation_and_inline_industry_filter(self):
        app = self.app()
        self.assertEqual(len(app.sidebar.selectbox), 0)
        self.assertEqual(app.selectbox(key='industry_filter').label,
                         '창업하고 싶은 업종을 선택하세요.')
        for consumer, key, tabs in [
            (True, 'consumer_tabs', ['상권 지도', '지역별 업종', '상세 데이터', '개요 요약']),
            (False, 'owner_tabs', ['상권 지도', '지표 비교', '상세 데이터', '개요 요약']),
        ]:
            for tab in tabs:
                app.button(key=f'nav_{key}_{tab}').click().run()
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertEqual(app.toggle(key='mode_switch').value, consumer)
                self.assertEqual(app.session_state[key], tab)
        app.selectbox(key='industry_filter').select_index(-1).run()
        self.assertFalse(app.exception)

    def test_launch_from_another_directory(self):
        original = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                self.assertEqual(len(self.app().dataframe[0].value), 10)
        finally:
            os.chdir(original)


if __name__ == '__main__':
    unittest.main()
