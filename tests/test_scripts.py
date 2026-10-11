import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'src' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScriptTests(unittest.TestCase):
    def test_imports_do_not_read_data_or_train_models(self):
        with patch('pandas.read_csv') as read, patch('joblib.dump') as dump:
            for path in sorted((ROOT / 'src').glob('[0-9]*.py')):
                module = load_script(path.name)
                self.assertTrue(callable(module.main))
                self.assertEqual(module.BASE_DIR, ROOT)
        read.assert_not_called()
        dump.assert_not_called()

    def test_constant_scores_and_missing_values(self):
        score = load_script('03_make_score.py').minmax_score
        self.assertEqual(score(pd.Series([5., 5., np.nan])).dropna().tolist(), [50., 50.])
        self.assertTrue(pd.isna(score(pd.Series([5., np.nan])).iloc[1]))
        self.assertEqual(score(pd.Series([10., 20., 30.])).tolist(), [0., 50., 100.])

    def test_time_evaluation_does_not_filter_future_outliers(self):
        module = load_script('11_time_test_model.py')
        rows = pd.DataFrame({
            '기준_년_코드': [2025] * 6,
            '기준_분기_코드': [1, 2, 3, 3, 4, 4],
            '총_유동인구_수': [100] * 6,
            '2030비율': [30] * 6,
            '점포_수': [5] * 6,
            '폐업비율': [2] * 6,
            '서비스_업종_코드_명': ['카페'] * 6,
            '상권_코드_명': ['테스트역'] * 6,
            '점포당_월매출': [10, 20, 30, 40, 1_000_000, 2_000_000],
        })
        with patch.object(module.pd, 'read_csv', return_value=rows), \
             patch.object(module, 'Pipeline') as pipeline, \
             patch.object(module.joblib, 'dump'), \
             patch.object(pd.DataFrame, 'to_csv'), redirect_stdout(io.StringIO()):
            pipeline.return_value.predict.return_value = np.array([30., 30.])
            module.main()
        fitted_target = pipeline.return_value.fit.call_args.args[1]
        self.assertEqual(fitted_target.tolist(), [10, 20, 30])
        self.assertEqual(len(pipeline.return_value.predict.call_args.args[0]), 2)


if __name__ == '__main__':
    unittest.main()
