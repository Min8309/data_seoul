"""Regression tests for lazy data loading and prediction cache invalidation."""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class LoadingTests(unittest.TestCase):
    def tearDown(self):
        st.cache_data.clear()
        st.cache_resource.clear()

    def test_owner_does_not_read_hotplace_csv(self):
        st.cache_data.clear()
        original = pd.read_csv
        paths = []

        def read(path, *args, **kwargs):
            paths.append(Path(path).name)
            return original(path, *args, **kwargs)

        with patch("pandas.read_csv", side_effect=read):
            app = AppTest.from_file(str(ROOT / "src/app.py"), default_timeout=60).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertIn("startup_ranking_map.csv", paths)
        self.assertNotIn("hotplace_score_map.csv", paths)

    def test_prediction_cache_reuses_inputs_and_invalidates_on_model_change(self):
        tree = ast.parse((ROOT / "src/app.py").read_text())
        function = next(n for n in tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == "predict_sales")
        model = Mock()
        model.predict.side_effect = lambda frame: frame["x"].to_numpy() * 2
        loader = Mock(return_value=model)
        namespace = {"st": st, "load_model": loader, "__name__": "loading_cache_test"}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(ROOT / "src/app.py"), "exec"), namespace)
        predict = namespace["predict_sales"]
        predict.clear()
        frame = pd.DataFrame({"x": [1.0, 2.0]})
        np.testing.assert_array_equal(predict(frame, 1), [2.0, 4.0])
        np.testing.assert_array_equal(predict(frame.copy(), 1), [2.0, 4.0])
        self.assertEqual(model.predict.call_count, 1)
        predict(frame, 2)
        self.assertEqual(model.predict.call_count, 2)
        predict(pd.DataFrame({"x": [3.0]}), 2)
        self.assertEqual(model.predict.call_count, 3)


if __name__ == "__main__":
    unittest.main()
