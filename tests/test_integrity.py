"""Meaningful edge checks for data handling, leakage boundaries and model IO."""

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import unquote
import numpy as np
import pandas as pd
from data_analysis.cleaning import clean_orders
from playbook.data import orders, tabular
from playbook.modeling import build_pipeline
from time_series.forecasting import features

ROOT = Path(__file__).resolve().parents[1]


class DataIntegrity(unittest.TestCase):
    def test_dirty_rows_are_auditable(self):
        valid, invalid, duplicates = clean_orders(orders())
        self.assertTrue(valid.order_id.is_unique)
        self.assertEqual(set(invalid.order_id), {3, 4, 6})
        self.assertEqual(set(duplicates.order_id), {2})
        self.assertEqual(len(valid) + len(invalid), orders().order_id.nunique())

    def test_future_values_do_not_change_past_features(self):
        raw = pd.DataFrame(
            {"date": pd.date_range("2025-01-01", periods=30), "y": np.arange(30.0)}
        )
        original = features(raw)
        raw.loc[raw.index >= 20, "y"] = 1e9
        changed = features(raw)
        pd.testing.assert_frame_equal(
            original.loc[original.index < 20], changed.loc[changed.index < 20]
        )
        self.assertEqual(original.loc[10, "rolling7"], np.arange(3, 10).mean())

    def test_unseen_category_and_all_empty_feature(self):
        x, y = tabular(n=100)
        x["empty"] = np.nan
        model = build_pipeline().fit(x, y)
        future = x.iloc[:3].copy()
        future["segment"] = "NEW"
        self.assertTrue(np.isfinite(model.predict(future)).all())

    def test_validation_does_not_change_imputer(self):
        x, y = tabular(n=100)
        model = build_pipeline().fit(x.iloc[:80], y.iloc[:80])
        imputer = (
            model.named_steps["prep"]
            .named_transformers_["numeric"]
            .named_steps["imputer"]
        )
        self.assertAlmostEqual(imputer.statistics_[0], x.iloc[:80].x0.median())
        before = imputer.statistics_.copy()
        shifted = x.iloc[80:].copy()
        shifted["x0"] = 1e12
        model.predict(shifted)
        np.testing.assert_equal(before, imputer.statistics_)

    def test_group_median_uses_training_only_and_fallback(self):
        from machine_learning.feature_engineering import GroupMedianImputer

        training = pd.DataFrame({"segment": ["A", "A", "B"], "x0": [1.0, 3.0, 10.0]})
        imputer = GroupMedianImputer().fit(training)
        future = pd.DataFrame(
            {"segment": ["A", "NEW", "B"], "x0": [np.nan, np.nan, 99.0]}
        )
        result = imputer.transform(future)
        np.testing.assert_allclose(result.x0, [2.0, 3.0, 99.0])
        self.assertTrue(future.x0.iloc[:2].isna().all())

    def test_profile_reports_missingness_and_numeric_range(self):
        from data_analysis.profile_csv import profile

        report = profile(
            pd.DataFrame({"x": [1.0, np.nan, 3.0], "kind": ["A", "A", "B"]})
        )
        self.assertAlmostEqual(report.loc["x", "missing_rate"], 1 / 3)
        self.assertEqual(report.loc["x", "numeric_median"], 2.0)
        self.assertEqual(report.loc["kind", "nunique"], 2)

    def test_relative_document_links(self):
        for path in ROOT.rglob("*.md"):
            if any(p in {".venv", ".git"} for p in path.parts):
                continue
            for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text()):
                if "://" in link or link.startswith("#"):
                    continue
                target = unquote(link.split("#")[0])
                self.assertTrue((path.parent / target).exists(), f"{path}: {link}")


class CLI(unittest.TestCase):
    def run_cli(self, *args, ok=True):
        run = subprocess.run(
            [sys.executable, "-m", "playbook.cli", *map(str, args)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if ok:
            self.assertEqual(run.returncode, 0, run.stderr)
        else:
            self.assertNotEqual(run.returncode, 0)
        return run

    def test_csv_train_predict_and_missing_column(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            x, y = tabular(n=120)
            labelled = x.assign(target=y, id=np.arange(len(x)))
            labelled.to_csv(folder / "train.csv", index=False)
            self.run_cli(
                "train",
                "--input",
                folder / "train.csv",
                "--target",
                "target",
                "--id-col",
                "id",
                "--output",
                folder / "model",
            )
            future = x.iloc[:7].assign(id=np.arange(7))
            future["segment"] = "UNSEEN"
            future.to_csv(folder / "test.csv", index=False)
            self.run_cli(
                "predict",
                "--input",
                folder / "test.csv",
                "--model",
                folder / "model/model.joblib",
                "--output",
                folder / "pred.csv",
            )
            result = pd.read_csv(folder / "pred.csv")
            self.assertEqual(result.id.tolist(), list(range(7)))
            self.assertTrue(np.isfinite(result.prediction).all())
            future.drop(columns="x1").to_csv(folder / "bad.csv", index=False)
            self.run_cli(
                "predict",
                "--input",
                folder / "bad.csv",
                "--model",
                folder / "model/model.joblib",
                ok=False,
            )

    def test_time_split_keeps_complete_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            x, y = tabular(n=120)
            x = x.assign(
                target=y,
                time=np.repeat(pd.date_range("2025-01-01", periods=30).astype(str), 4),
            )
            x.to_csv(folder / "train.csv", index=False)
            self.run_cli(
                "train",
                "--input",
                folder / "train.csv",
                "--target",
                "target",
                "--time-col",
                "time",
                "--output",
                folder / "model",
            )
            report = json.loads((folder / "model/metrics.json").read_text())
            self.assertEqual(report["train_rows"], 96)
            self.assertEqual(report["valid_rows"], 24)
            predictions = pd.read_csv(folder / "model/validation_predictions.csv")
            self.assertEqual(predictions.source_row.min(), 96)

    def test_group_split_has_disjoint_entities(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            x, y = tabular(n=120)
            x = x.assign(target=y, group=np.repeat(np.arange(30), 4))
            x.to_csv(folder / "train.csv", index=False)
            self.run_cli(
                "train",
                "--input",
                folder / "train.csv",
                "--target",
                "target",
                "--group-col",
                "group",
                "--output",
                folder / "model",
            )
            b = pd.read_csv(
                folder / "model/validation_predictions.csv"
            ).source_row.to_numpy()
            a = np.setdiff1d(np.arange(len(x)), b)
            self.assertTrue(set(x.iloc[a].group).isdisjoint(x.iloc[b].group))

    def test_classification_probability_output(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            x, y = tabular("classification", n=120)
            x.assign(target=y).to_csv(folder / "train.csv", index=False)
            x.iloc[:5].to_csv(folder / "test.csv", index=False)
            self.run_cli(
                "train",
                "--input",
                folder / "train.csv",
                "--target",
                "target",
                "--task",
                "classification",
                "--output",
                folder / "model",
            )
            self.run_cli(
                "predict",
                "--input",
                folder / "test.csv",
                "--model",
                folder / "model/model.joblib",
                "--output",
                folder / "pred.csv",
            )
            pred = pd.read_csv(folder / "pred.csv").filter(like="probability_")
            self.assertEqual(pred.shape, (5, 2))
            np.testing.assert_allclose(pred.sum(axis=1), 1)


if __name__ == "__main__":
    unittest.main()
