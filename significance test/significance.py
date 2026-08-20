#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outputs
-------
00_input_file_manifest.csv
01_raw_all_runs_long.csv
02_all_method_summary.csv
03_primary_strongest_baseline_tests_full.csv
04_all_pairwise_tests_full.csv
05_primary_table_ready.csv
06_all_pairwise_table_ready.csv
07_data_quality_report.csv

python .\significance.py `
  --deepddi ".\deepddi_all.csv" `
  --drugdagt ".\drugdagt_all.csv" `
  --kgnn ".\kgnn_all.csv" `
  --node2vec ".\node2vec_all.csv" `
  --skipgnn ".\skipgnn_all.csv" `
  --sumgnn ".\sumgnn_all.csv" `
  --raddi ".\raddi_all.csv" `
  --out_dir ".\statistical_analysis"
"""

import argparse
import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


DATASETS = [
    "drugbank_3class",
    "ddinter_4class",
    "pdd_graph_2class",
]
DATASET_DISPLAY = {
    "drugbank_3class": "DrugBank",
    "ddinter_4class": "DDInter",
    "pdd_graph_2class": "PDD-Graph",
}
METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "Macro-F1",
    "AUC-ROC",
    "AUC-PR",
]
METHOD_ORDER = [
    "DeepDDI",
    "DrugDAGT",
    "Node2Vec+MLP",
    "SkipGNN",
    "KGNN-DDI",
    "SumGNN",
    "RA-DDI",
]
BASELINES = [m for m in METHOD_ORDER if m != "RA-DDI"]
EXPECTED_KEYS = {(seed, fold) for seed in (42, 43, 44) for fold in range(1, 6)}

METHOD_SCHEMAS = {
    "DeepDDI": {
        "dataset": ["Dataset", "dataset"],
        "seed": ["SplitSeed", "seed", "Seed"],
        "fold": ["Fold", "fold"],
        "metrics": {
            "Accuracy": ["accuracy", "Accuracy", "test_accuracy"],
            "Precision": ["precision_macro", "Precision", "test_precision_macro"],
            "Recall": ["recall_macro", "Recall", "test_recall_macro"],
            "Macro-F1": ["f1_macro", "Macro-F1", "macro_f1", "test_f1_macro"],
            "AUC-ROC": ["roc_auc_macro_ovr", "AUC-ROC", "auc_roc"],
            "AUC-PR": ["auprc_macro", "AUC-PR", "auc_pr"],
        },
    },
    "DrugDAGT": {
        "dataset": ["Dataset", "dataset"],
        "seed": ["SplitSeed", "seed", "Seed"],
        "fold": ["Fold", "fold"],
        "metrics": {m: [m] for m in METRICS},
    },
    "KGNN-DDI": {
        "dataset": ["dataset", "Dataset"],
        "seed": ["seed", "SplitSeed", "Seed"],
        "fold": ["fold", "Fold"],
        "metrics": {
            "Accuracy": ["acc"],
            "Precision": ["precision_macro"],
            "Recall": ["recall_macro"],
            "Macro-F1": ["f1_macro"],
            "AUC-ROC": ["auroc_macro"],
            "AUC-PR": ["aupr_macro"],
        },
    },
    "Node2Vec+MLP": {
        "dataset": ["dataset", "Dataset"],
        "seed": ["seed", "SplitSeed", "Seed"],
        "fold": ["fold", "Fold"],
        "metrics": {
            "Accuracy": ["accuracy"],
            "Precision": ["precision_macro"],
            "Recall": ["recall_macro"],
            "Macro-F1": ["f1_macro"],
            "AUC-ROC": ["roc_auc_macro_ovr"],
            "AUC-PR": ["aupr_macro_ovr"],
        },
    },
    "SkipGNN": {
        "dataset": ["dataset", "Dataset"],
        "seed": ["seed", "SplitSeed", "Seed"],
        "fold": ["fold", "Fold"],
        "metrics": {
            "Accuracy": ["accuracy"],
            "Precision": ["macro_precision"],
            "Recall": ["macro_recall"],
            "Macro-F1": ["macro_f1"],
            "AUC-ROC": ["auc_roc"],
            "AUC-PR": ["auc_pr"],
        },
    },
    "SumGNN": {
        "dataset": ["Dataset", "dataset"],
        "seed": ["SplitSeed", "seed", "Seed"],
        "fold": ["Fold", "fold"],
        "metrics": {m: [m] for m in METRICS},
    },
    "RA-DDI": {
        "dataset": ["Dataset", "dataset"],
        "seed": ["seed", "SplitSeed", "Seed"],
        "fold": ["Fold", "fold"],
        "metrics": {m: [m] for m in METRICS},
    },
}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]
    return df


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    magic = path.read_bytes()[:8]
    is_ole_xls = magic.startswith(b"\xD0\xCF\x11\xE0")
    is_zip_excel = magic.startswith(b"PK\x03\x04")

    if is_ole_xls or is_zip_excel or path.suffix.lower() in {".xls", ".xlsx"}:
        try:
            sheets = pd.read_excel(path, sheet_name=None)
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is an Excel workbook, not a true CSV. "
                "For legacy .xls files install xlrd with "
                "`python -m pip install xlrd`, or save the file as CSV UTF-8 "
                "in Excel and rerun."
            ) from exc
        frames = []
        for sheet_name, frame in sheets.items():
            if frame is None or frame.empty:
                continue
            frame = clean_columns(frame)
            frame["_SourceSheet"] = sheet_name
            frames.append(frame)
        if not frames:
            raise ValueError(f"No non-empty sheets found in {path}")
        return pd.concat(frames, ignore_index=True)

    errors = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin1"):
        try:
            return clean_columns(pd.read_csv(path, encoding=encoding))
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError(
        f"Could not read {path} as CSV. Attempts:\n" + "\n".join(errors)
    )


def pick_column(df: pd.DataFrame, candidates: Iterable[str], field: str) -> str:
    exact = {c: c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise KeyError(
        f"Cannot identify {field}. Candidates={list(candidates)}; "
        f"available columns={list(df.columns)}"
    )


def normalize_dataset_name(value: object) -> str:
    s = str(value).replace("\ufeff", "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")

    aliases = {
        "drugbank_3class": "drugbank_3class",
        "drugbank": "drugbank_3class",
        "ddinter_4class": "ddinter_4class",
        "ddinter": "ddinter_4class",
        "pdd_graph_2class": "pdd_graph_2class",
        "pdd_graph": "pdd_graph_2class",
        "pddgraph": "pdd_graph_2class",
    }

    if s not in aliases:
        raise ValueError(
            f"Unexpected dataset label: {value!r}. "
            f"Accepted labels/aliases: {sorted(aliases)}"
        )
    return aliases[s]


def normalize_method(method: str, path: Path) -> Tuple[pd.DataFrame, List[str]]:
    raw = read_table(path)
    schema = METHOD_SCHEMAS[method]
    report = []

    dataset_col = pick_column(raw, schema["dataset"], "dataset column")
    seed_col = pick_column(raw, schema["seed"], "seed column")
    fold_col = pick_column(raw, schema["fold"], "fold column")

    out = pd.DataFrame(index=raw.index.copy())
    out["SourceRow"] = np.arange(len(raw), dtype=np.int64)

    raw_dataset = raw[dataset_col].astype(str).str.strip()
    out["Dataset"] = raw_dataset.map(normalize_dataset_name)
    out["Seed"] = pd.to_numeric(raw[seed_col], errors="coerce")
    out["Fold"] = pd.to_numeric(raw[fold_col], errors="coerce")
    out["Method"] = method
    out["SourceFile"] = str(path.resolve())

    metric_source_columns = {}

    for metric, candidates in schema["metrics"].items():
        col = pick_column(raw, candidates, f"{metric} column")
        metric_source_columns[metric] = col

        source_numeric = pd.to_numeric(raw[col], errors="coerce")
        out[metric] = source_numeric

        # Conversion is not allowed to alter metric values.
        src_arr = source_numeric.to_numpy(dtype=float)
        out_arr = out[metric].to_numpy(dtype=float)
        same = np.isclose(
            src_arr,
            out_arr,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
        if not bool(np.all(same)):
            bad_idx = np.where(~same)[0][:10]
            details = [
                f"row={int(i)}, source={src_arr[i]!r}, converted={out_arr[i]!r}"
                for i in bad_idx
            ]
            raise RuntimeError(
                f"{method}: metric fidelity check failed for {metric} "
                f"(source column {col!r}). Examples: "
                + "; ".join(details)
            )

    out["Seed"] = out["Seed"].astype("Int64")
    out["Fold"] = out["Fold"].astype("Int64")

    mapping_text = ", ".join(
        f"{metric}<-{column}"
        for metric, column in metric_source_columns.items()
    )
    report.append(
        f"[INFO] {method}: source columns: dataset={dataset_col}, "
        f"seed={seed_col}, fold={fold_col}; metrics: {mapping_text}"
    )

    return out.reset_index(drop=True), report


def validate_data(data: pd.DataFrame) -> Tuple[List[str], bool]:
    lines = []
    fatal = False

    lines.append("=== Statistical analysis input validation ===")
    lines.append(f"Total normalized rows: {len(data)}")
    lines.append(
        "Expected complete design: "
        f"{len(METHOD_ORDER)} methods x {len(DATASETS)} datasets x "
        f"{len(EXPECTED_KEYS)} runs = "
        f"{len(METHOD_ORDER) * len(DATASETS) * len(EXPECTED_KEYS)} rows"
    )

    for method in METHOD_ORDER:
        for dataset in DATASETS:
            sub = data[
                (data["Method"] == method)
                & (data["Dataset"] == dataset)
            ].copy()

            keys = {
                (int(seed), int(fold))
                for seed, fold in zip(sub["Seed"], sub["Fold"])
                if pd.notna(seed) and pd.notna(fold)
            }
            missing = sorted(EXPECTED_KEYS - keys)
            extra = sorted(keys - EXPECTED_KEYS)
            duplicate_count = int(
                sub.duplicated(["Dataset", "Seed", "Fold"]).sum()
            )
            metric_nan = int(sub[METRICS].isna().sum().sum())

            status = "OK"
            if (
                len(sub) != 15
                or missing
                or extra
                or duplicate_count
                or metric_nan
            ):
                status = "ERROR"
                fatal = True

            lines.append(
                f"[{status}] {method:13s} | {dataset:18s} | "
                f"rows={len(sub):2d} | missing_keys={missing} | "
                f"extra_keys={extra} | duplicate_keys={duplicate_count} | "
                f"missing_metric_cells={metric_nan}"
            )

            if len(sub):
                outside = {}
                for metric in METRICS:
                    bad = sub[(sub[metric] < 0) | (sub[metric] > 1)]
                    if len(bad):
                        outside[metric] = len(bad)
                if outside:
                    fatal = True
                    lines.append(
                        f"  [ERROR] values outside [0,1]: {outside}"
                    )

    metric_dups = data[
        data.duplicated(
            ["Method", "Dataset"] + METRICS,
            keep=False,
        )
    ].sort_values(["Method", "Dataset"] + METRICS)
    if len(metric_dups):
        lines.append(
            "\n[WARNING] Exact duplicate six-metric vectors were found across "
            "different split keys. Verify that these are not copy/paste errors:"
        )
        lines.append(
            metric_dups[
                ["Method", "Dataset", "Seed", "Fold"] + METRICS
            ].to_string(index=False)
        )

    for dataset in DATASETS:
        method_keys = {}
        for method in METHOD_ORDER:
            sub = data[
                (data["Method"] == method)
                & (data["Dataset"] == dataset)
            ]
            method_keys[method] = {
                (int(s), int(f))
                for s, f in zip(sub["Seed"], sub["Fold"])
                if pd.notna(s) and pd.notna(f)
            }
        reference = method_keys["RA-DDI"]
        for method, keys in method_keys.items():
            if keys != reference:
                fatal = True
                lines.append(
                    f"[ERROR] {dataset}: matched split keys differ between "
                    f"RA-DDI and {method}. "
                    f"RA-only={sorted(reference - keys)}, "
                    f"{method}-only={sorted(keys - reference)}"
                )

    lines.append(
        "\nDependence note: the 15 matched split-level observations are not "
        "strictly independent because folds within repeated cross-validation "
        "share partially overlapping training data."
    )
    return lines, fatal


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for i, value in enumerate(sorted_p):
        adjusted = (m - i) * value
        running_max = max(running_max, adjusted)
        adjusted_sorted[i] = min(1.0, running_max)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def rank_biserial(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isclose(diff, 0.0, atol=1e-15)]
    if len(diff) == 0:
        return 0.0
    ranks = rankdata(np.abs(diff), method="average")
    pos = float(ranks[diff > 0].sum())
    neg = float(ranks[diff < 0].sum())
    denom = pos + neg
    return 0.0 if denom == 0 else (pos - neg) / denom


def paired_test(
    ra: pd.DataFrame,
    baseline: pd.DataFrame,
    metric: str,
) -> Dict[str, object]:
    keys = ["Dataset", "Seed", "Fold"]
    left = ra[keys + [metric]].rename(columns={metric: "RA"})
    right = baseline[keys + [metric]].rename(columns={metric: "Baseline"})
    merged = left.merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
    ).sort_values(["Seed", "Fold"])

    if len(merged) != 15:
        raise ValueError(
            f"Expected 15 matched observations, found {len(merged)}."
        )

    diff = merged["RA"].to_numpy(float) - merged["Baseline"].to_numpy(float)
    if np.all(np.isclose(diff, 0.0, atol=1e-15)):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(
            merged["RA"].to_numpy(float),
            merged["Baseline"].to_numpy(float),
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="auto",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    wins = int(np.sum(diff > 1e-15))
    losses = int(np.sum(diff < -1e-15))
    ties = int(len(diff) - wins - losses)

    return {
        "N": len(merged),
        "RA mean": float(merged["RA"].mean()),
        "RA std": float(merged["RA"].std(ddof=1)),
        "Baseline mean": float(merged["Baseline"].mean()),
        "Baseline std": float(merged["Baseline"].std(ddof=1)),
        "Mean paired difference": float(diff.mean()),
        "Median paired difference": float(np.median(diff)),
        "Wins": wins,
        "Ties": ties,
        "Losses": losses,
        "W/T/L": f"{wins}/{ties}/{losses}",
        "Wilcoxon statistic": statistic,
        "Raw p": p_value,
        "Rank-biserial r": rank_biserial(diff),
    }


def significance_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for method in METHOD_ORDER:
            sub = data[
                (data["Dataset"] == dataset)
                & (data["Method"] == method)
            ]
            row = {
                "Dataset": dataset,
                "Dataset display": DATASET_DISPLAY[dataset],
                "Method": method,
                "N": len(sub),
            }
            for metric in METRICS:
                row[f"{metric} mean"] = float(sub[metric].mean())
                row[f"{metric} std"] = float(sub[metric].std(ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


def build_primary_tests(
    data: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        ra = data[
            (data["Dataset"] == dataset)
            & (data["Method"] == "RA-DDI")
        ]
        for metric in METRICS:
            baseline_summary = summary[
                (summary["Dataset"] == dataset)
                & (summary["Method"].isin(BASELINES))
            ].copy()

            best_value = baseline_summary[f"{metric} mean"].max()
            tied = baseline_summary[
                np.isclose(
                    baseline_summary[f"{metric} mean"],
                    best_value,
                    atol=1e-12,
                    rtol=0,
                )
            ].sort_values("Method")
            best_method = tied.iloc[0]["Method"]

            baseline = data[
                (data["Dataset"] == dataset)
                & (data["Method"] == best_method)
            ]
            stats = paired_test(ra, baseline, metric)
            rows.append({
                "Dataset": dataset,
                "Dataset display": DATASET_DISPLAY[dataset],
                "Metric": metric,
                "Baseline": best_method,
                "Baseline selection": (
                    "Highest 15-run baseline mean for this dataset/metric"
                ),
                **stats,
            })

    out = pd.DataFrame(rows)
    out["Holm p"] = holm_adjust(out["Raw p"])
    out["Significance"] = out["Holm p"].map(significance_label)
    out["Direction"] = np.where(
        out["Mean paired difference"] > 0,
        "RA-DDI > baseline",
        np.where(
            out["Mean paired difference"] < 0,
            "RA-DDI < baseline",
            "Equal",
        ),
    )
    return out


def build_all_pairwise_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        ra = data[
            (data["Dataset"] == dataset)
            & (data["Method"] == "RA-DDI")
        ]
        for baseline_method in BASELINES:
            baseline = data[
                (data["Dataset"] == dataset)
                & (data["Method"] == baseline_method)
            ]
            for metric in METRICS:
                stats = paired_test(ra, baseline, metric)
                rows.append({
                    "Dataset": dataset,
                    "Dataset display": DATASET_DISPLAY[dataset],
                    "Baseline": baseline_method,
                    "Metric": metric,
                    **stats,
                })

    out = pd.DataFrame(rows)
    out["Holm p"] = holm_adjust(out["Raw p"])
    out["Significance"] = out["Holm p"].map(significance_label)
    out["Direction"] = np.where(
        out["Mean paired difference"] > 0,
        "RA-DDI > baseline",
        np.where(
            out["Mean paired difference"] < 0,
            "RA-DDI < baseline",
            "Equal",
        ),
    )
    return out



def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def format_p(value: float) -> str:
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def build_primary_table_ready(primary: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Dataset": primary["Dataset display"],
        "Metric": primary["Metric"],
        "Baseline": primary["Baseline"],
        "Baseline mean ± std": [
            format_mean_std(m, s)
            for m, s in zip(
                primary["Baseline mean"],
                primary["Baseline std"],
            )
        ],
        "RA-DDI mean ± std": [
            format_mean_std(m, s)
            for m, s in zip(
                primary["RA mean"],
                primary["RA std"],
            )
        ],
        "Mean paired difference": primary["Mean paired difference"],
        "Wins/Ties/Losses": primary["W/T/L"],
        "Wilcoxon raw p": primary["Raw p"],
        "Wilcoxon raw p (formatted)": primary["Raw p"].map(format_p),
        "Holm-adjusted p": primary["Holm p"],
        "Holm-adjusted p (formatted)": primary["Holm p"].map(format_p),
        "Rank-biserial r": primary["Rank-biserial r"],
        "Sig.": primary["Significance"],
        "Direction": primary["Direction"],
        "N matched runs": primary["N"],
    })
    return out


def build_all_pairwise_table_ready(all_pairwise: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Dataset": all_pairwise["Dataset display"],
        "Baseline": all_pairwise["Baseline"],
        "Metric": all_pairwise["Metric"],
        "Baseline mean ± std": [
            format_mean_std(m, s)
            for m, s in zip(
                all_pairwise["Baseline mean"],
                all_pairwise["Baseline std"],
            )
        ],
        "RA-DDI mean ± std": [
            format_mean_std(m, s)
            for m, s in zip(
                all_pairwise["RA mean"],
                all_pairwise["RA std"],
            )
        ],
        "Mean paired difference": all_pairwise["Mean paired difference"],
        "Wins/Ties/Losses": all_pairwise["W/T/L"],
        "Wilcoxon raw p": all_pairwise["Raw p"],
        "Wilcoxon raw p (formatted)": all_pairwise["Raw p"].map(format_p),
        "Holm-adjusted p": all_pairwise["Holm p"],
        "Holm-adjusted p (formatted)": all_pairwise["Holm p"].map(format_p),
        "Rank-biserial r": all_pairwise["Rank-biserial r"],
        "Sig.": all_pairwise["Significance"],
        "Direction": all_pairwise["Direction"],
        "N matched runs": all_pairwise["N"],
    })
    return out


def build_quality_report_csv(
    data: pd.DataFrame,
    normalization_messages: List[str],
) -> pd.DataFrame:
    rows = []

    for message in normalization_messages:
        rows.append({
            "Level": "INFO",
            "Method": "",
            "Dataset": "",
            "Check": "Dataset label normalization",
            "Details": message.replace("\n", " | "),
        })

    expected_total = (
        len(METHOD_ORDER) * len(DATASETS) * len(EXPECTED_KEYS)
    )
    rows.append({
        "Level": "INFO",
        "Method": "",
        "Dataset": "",
        "Check": "Expected complete design",
        "Details": (
            f"{len(METHOD_ORDER)} methods × {len(DATASETS)} datasets × "
            f"{len(EXPECTED_KEYS)} matched runs = {expected_total} rows"
        ),
    })

    for method in METHOD_ORDER:
        for dataset in DATASETS:
            sub = data[
                (data["Method"] == method)
                & (data["Dataset"] == dataset)
            ].copy()

            keys = {
                (int(seed), int(fold))
                for seed, fold in zip(sub["Seed"], sub["Fold"])
                if pd.notna(seed) and pd.notna(fold)
            }
            missing = sorted(EXPECTED_KEYS - keys)
            extra = sorted(keys - EXPECTED_KEYS)
            duplicate_count = int(
                sub.duplicated(["Dataset", "Seed", "Fold"]).sum()
            )
            metric_nan = int(sub[METRICS].isna().sum().sum())

            level = "PASS"
            if (
                len(sub) != 15
                or missing
                or extra
                or duplicate_count
                or metric_nan
            ):
                level = "ERROR"

            rows.append({
                "Level": level,
                "Method": method,
                "Dataset": DATASET_DISPLAY.get(dataset, dataset),
                "Check": "15 matched seed-fold runs",
                "Details": (
                    f"rows={len(sub)}; missing_keys={missing}; "
                    f"extra_keys={extra}; duplicate_keys={duplicate_count}; "
                    f"missing_metric_cells={metric_nan}"
                ),
            })

    metric_dups = data[
        data.duplicated(
            ["Method", "Dataset"] + METRICS,
            keep=False,
        )
    ].sort_values(["Method", "Dataset", "Seed", "Fold"])

    if len(metric_dups):
        for _, row in metric_dups.iterrows():
            rows.append({
                "Level": "WARNING",
                "Method": row["Method"],
                "Dataset": DATASET_DISPLAY.get(
                    row["Dataset"], row["Dataset"]
                ),
                "Check": "Exact duplicate metric vector",
                "Details": (
                    f"Seed={row['Seed']}, Fold={row['Fold']}; "
                    + ", ".join(
                        f"{m}={row[m]:.6f}" for m in METRICS
                    )
                ),
            })

    rows.append({
        "Level": "INFO",
        "Method": "",
        "Dataset": "",
        "Check": "Dependence structure",
        "Details": (
            "The statistical unit is one matched Dataset-Seed-Fold run. "
            "Repeated-CV folds contain partially overlapping training sets, "
            "so the 15 paired observations are not strictly independent."
        ),
    })

    return pd.DataFrame(rows)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_input_manifest(paths: Dict[str, Path]) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        path = paths[method]
        rows.append({
            "Method": method,
            "SourceFile": str(path.resolve()),
            "SizeBytes": path.stat().st_size,
            "SHA256": sha256_file(path),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepddi", required=True)
    parser.add_argument("--drugdagt", required=True)
    parser.add_argument("--kgnn", required=True)
    parser.add_argument("--node2vec", required=True)
    parser.add_argument("--skipgnn", required=True)
    parser.add_argument("--sumgnn", required=True)
    parser.add_argument("--raddi", required=True)
    parser.add_argument("--out_dir", default="statistical_analysis")
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only normalize and validate input files; do not run tests.",
    )
    args = parser.parse_args()

    paths = {
        "DeepDDI": Path(args.deepddi),
        "DrugDAGT": Path(args.drugdagt),
        "KGNN-DDI": Path(args.kgnn),
        "Node2Vec+MLP": Path(args.node2vec),
        "SkipGNN": Path(args.skipgnn),
        "SumGNN": Path(args.sumgnn),
        "RA-DDI": Path(args.raddi),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    normalization_messages = []
    for method in METHOD_ORDER:
        frame, messages = normalize_method(method, paths[method])
        frames.append(frame)
        normalization_messages.extend(messages)

    data = pd.concat(frames, ignore_index=True)

    unexpected = sorted(set(data["Dataset"]) - set(DATASETS))
    if unexpected:
        raise RuntimeError(
            f"Unexpected datasets after normalization: {unexpected}"
        )

    input_manifest = build_input_manifest(paths)
    input_manifest.to_csv(
        out_dir / "00_input_file_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    data["Seed"] = data["Seed"].astype(int)
    data["Fold"] = data["Fold"].astype(int)
    data = data.sort_values(
        ["Dataset", "Method", "Seed", "Fold"]
    ).reset_index(drop=True)

    validation_lines, fatal = validate_data(data)

    data.to_csv(
        out_dir / "01_raw_all_runs_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    quality_table = build_quality_report_csv(
        data,
        normalization_messages,
    )
    quality_path = out_dir / "07_data_quality_report.csv"
    quality_table.to_csv(
        quality_path,
        index=False,
        encoding="utf-8-sig",
    )

    if fatal:
        print("\n".join(normalization_messages + validation_lines))
        raise SystemExit(
            "\nInput validation failed. No significance tests were run. "
            f"See {quality_path}"
        )

    if args.validate_only:
        print(f"Validation passed. Report: {quality_path}")
        return

    summary = build_summary(data)
    primary = build_primary_tests(data, summary)
    all_pairwise = build_all_pairwise_tests(data)

    summary.to_csv(
        out_dir / "02_all_method_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary.to_csv(
        out_dir / "03_primary_strongest_baseline_tests_full.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_pairwise.to_csv(
        out_dir / "04_all_pairwise_tests_full.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary_table = build_primary_table_ready(primary)
    pairwise_table = build_all_pairwise_table_ready(all_pairwise)
    quality_table = build_quality_report_csv(
        data,
        normalization_messages,
    )

    primary_table.to_csv(
        out_dir / "05_primary_table_ready.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pairwise_table.to_csv(
        out_dir / "06_all_pairwise_table_ready.csv",
        index=False,
        encoding="utf-8-sig",
    )
    quality_table.to_csv(
        out_dir / "07_data_quality_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Statistical analysis completed.")
    print(f"Output directory: {out_dir.resolve()}")
    print(
        "Primary comparisons: "
        f"{len(primary)}; all pairwise comparisons: {len(all_pairwise)}"
    )


if __name__ == "__main__":
    main()
