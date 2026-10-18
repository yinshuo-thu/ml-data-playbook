"""Run offline examples in separate processes, capture logs and fail on errors."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

CORE = [
    "numpy_basics.arrays",
    "data_analysis.cleaning",
    "data_analysis.groupby_reshape",
    "data_analysis.joins",
    "data_analysis.io_memory",
    "data_analysis.dates_strings",
    "visualization.eda_report",
    "statistical_analysis.bootstrap_ab",
    "sql.sqlite_analysis",
    "machine_learning.regression",
    "machine_learning.classification",
    "machine_learning.tuning",
    "machine_learning.feature_engineering",
    "machine_learning.feature_importance",
    "machine_learning.text_classification",
    "machine_learning.clustering_pca",
    "machine_learning.validation",
    "time_series.forecasting",
    "scripts.make_demo_data",
    "data_analysis.profile_csv",
]
DEEP = [
    "deep_learning.tensors",
    "deep_learning.mlp",
    "deep_learning.cnn",
    "deep_learning.sequence",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["core", "deep"], default="core")
    args = p.parse_args()
    out = Path("outputs/smoke_logs")
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for module in CORE if args.suite == "core" else DEEP:
        start = time.monotonic()
        command = [sys.executable, "-m", module]
        if module == "data_analysis.profile_csv":
            command += ["--input", "data/demo/regression_train.csv"]
        run = subprocess.run(command, capture_output=True, text=True, timeout=240)
        (out / f"{module}.log").write_text(run.stdout + run.stderr)
        report.append(
            {
                "module": module,
                "returncode": run.returncode,
                "seconds": round(time.monotonic() - start, 2),
            }
        )
        print(module, "PASS" if run.returncode == 0 else "FAIL", flush=True)
    Path(f"outputs/smoke_{args.suite}.json").write_text(json.dumps(report, indent=2))
    if any(item["returncode"] != 0 for item in report):
        raise SystemExit("Example failed; inspect outputs/smoke_logs")


if __name__ == "__main__":
    main()
