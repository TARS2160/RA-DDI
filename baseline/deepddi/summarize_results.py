from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

DEFAULT_METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_micro", "f1_weighted", "roc_auc_macro_ovr", "auprc_macro"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True, help="Directory containing per-fold metrics.csv files, e.g. strict_results/DeepDDI")
    ap.add_argument("--out_csv", required=True, help="Output summary CSV path")
    ap.add_argument("--all_csv", default=None, help="Optional concatenated metrics CSV path")
    args = ap.parse_args()

    files = sorted(Path(args.results_root).rglob("metrics.csv"))
    if not files:
        raise SystemExit(f"No metrics.csv found under {args.results_root}")
    all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if args.all_csv:
        Path(args.all_csv).parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(args.all_csv, index=False)

    metric_cols = [m for m in DEFAULT_METRICS if m in all_df.columns]
    rows = []
    for (dataset, method), g in all_df.groupby(["dataset", "method"], dropna=False):
        row = {"dataset": dataset, "method": method, "n_runs": len(g)}
        for m in metric_cols:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1)
            row[f"{m}_mean_std"] = f"{g[m].mean():.4f} ± {g[m].std(ddof=1):.4f}"
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["dataset", "method"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved summary: {args.out_csv}")

if __name__ == "__main__":
    main()
