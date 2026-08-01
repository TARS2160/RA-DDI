# -*- coding: utf-8 -*-
"""
Generate shared train/val/test splits for DDI baseline experiments.

Leakage-safe protocol:
- Canonicalize every pair so that (A, B) and (B, A) share one pair_group.
- Outer split: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).
- Inner validation split: StratifiedGroupKFold(n_splits=8, shuffle=True,
  random_state=seed + fold), taking one inner fold as validation.
  => approximately 70% train, 10% validation, and 20% test, while keeping
     every unordered drug pair in exactly one partition.
- Assert and report zero pair_group overlap among train, validation, and test.

The script writes:
1) prepared/<dataset>_prepared.csv
   Filtered dataset with standardized columns and label_id.
2) splits/<dataset>/seed_<seed>_fold_<fold>.npz
   Compact numpy arrays: train_idx, val_idx, test_idx.
3) split_manifest.csv
   One row per dataset/seed/fold with class distributions and file paths.
4) dataset_summary.csv
   Dataset sizes, label counts, number of drugs/pairs.
5) method_input_setting_template.csv
   A table template for manuscript reporting.

Usage:
    python generate_shared_splits_group_safe.py --out_dir results/shared_protocol_group

This upload version always uses group-aware splitting; there is no unsafe
row-wise fallback.

Edit DATASETS below if your local paths differ.
"""

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


DATASETS = {
    "drugbank_3class": {
        "path": "D:/study/gnn/DDKG-main/DDKG-main/data/drugbank/test.csv",
        "drug1_col": "drug1_id",
        "drug2_col": "drug2_id",
        "label_col": "label",
        "label_order": ["increase", "decrease", "adverse"],
        "task": "3-class DDI relation classification",
        "negative_sampling": "N/A; all samples are known typed DDI pairs",
    },
    "ogbl_ddi_derived_4class": {
        "path": "D:/study/gnn/DDKG-main/DDKG-main/data/ogb_ddi/ogbl_ddi/mapping/test1.csv",
        "drug1_col": "first drug id",
        "drug2_col": "second drug id",
        "label_col": "label",
        # Raw file contains a rare 'interaction' label. For the user's 4-class setting, it is excluded.
        "label_order": ["increase", "decrease", "adverse", "unknown"],
        "task": "4-class derived DDI relation classification",
        "negative_sampling": "N/A for this derived multi-class setting",
    },
    "pdd_graph_2class": {
        "path": "D:/study/gnn/DDKG-main/DDKG-main/data/PDD_graph/test2.csv",
        "drug1_col": "first drug id",
        "drug2_col": "second drug id",
        "label_col": "label",
        "label_order": [0, 1],
        "task": "2-class relation/link prediction",
        "negative_sampling": "Input file already contains positives and negatives; treat it as the fixed shared sample set",
    },
    "ddinter_4class": {
        "path": "D:/study/gnn/DDKG-main/DDKG-main/data/ddinter/test3.csv",
        "drug1_col": "first drug id",
        "drug2_col": "second drug id",
        "label_col": "label",
        "label_order": ["Minor", "Moderate", "Major", "Unknown"],
        "task": "4-class DDInter severity/relation classification",
        "negative_sampling": "N/A; all samples have curated interaction labels",
    },
}


def canon_pair(a: Any, b: Any) -> str:
    a, b = str(a).strip(), str(b).strip()
    return "||".join(sorted([a, b]))


def safe_count(indices: np.ndarray, labels: np.ndarray) -> str:
    cnt = Counter(labels[indices].tolist())
    return json.dumps({str(k): int(v) for k, v in sorted(cnt.items())}, ensure_ascii=False)


def prepare_dataset(name: str, cfg: Dict[str, Any], prepared_dir: str):
    df = pd.read_csv(cfg["path"])
    d1, d2, lbl = cfg["drug1_col"], cfg["drug2_col"], cfg["label_col"]

    missing = [c for c in [d1, d2, lbl] if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}; available={df.columns.tolist()}")

    df = df.copy()
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    df[d1] = df[d1].astype(str).str.strip()
    df[d2] = df[d2].astype(str).str.strip()

    label_order = cfg["label_order"]
    label_map = {lab: i for i, lab in enumerate(label_order)}
    label_map_str = {str(lab): i for i, lab in enumerate(label_order)}

    def map_label(x):
        if x in label_map:
            return label_map[x]
        sx = str(x)
        if sx in label_map_str:
            return label_map_str[sx]
        return np.nan

    df["label_raw"] = df[lbl]
    df["label_id"] = df[lbl].map(map_label)

    before = len(df)
    dropped_labels = df[df["label_id"].isna()][lbl].value_counts(dropna=False).to_dict()
    df = df[~df["label_id"].isna()].copy()
    df["label_id"] = df["label_id"].astype(int)

    # Standardized columns used by all baselines.
    df["drug1_id_std"] = df[d1].astype(str)
    df["drug2_id_std"] = df[d2].astype(str)
    df["pair_group"] = [canon_pair(a, b) for a, b in zip(df["drug1_id_std"], df["drug2_id_std"])]
    df["sample_id"] = np.arange(len(df), dtype=np.int64)

    keep_front = ["sample_id", "row_id", "drug1_id_std", "drug2_id_std", "label_raw", "label_id", "pair_group"]
    other_cols = [c for c in df.columns if c not in keep_front]
    df = df[keep_front + other_cols]

    os.makedirs(prepared_dir, exist_ok=True)
    out_path = os.path.join(prepared_dir, f"{name}_prepared.csv")
    df.to_csv(out_path, index=False)

    summary = {
        "Dataset": name,
        "Task": cfg["task"],
        "RawRows": before,
        "PreparedRows": len(df),
        "DroppedRows": before - len(df),
        "DroppedLabelCounts": json.dumps({str(k): int(v) for k, v in dropped_labels.items()}, ensure_ascii=False),
        "LabelOrder": json.dumps([str(x) for x in label_order], ensure_ascii=False),
        "LabelCounts": json.dumps({str(k): int(v) for k, v in df["label_raw"].value_counts().to_dict().items()}, ensure_ascii=False),
        "LabelIdCounts": json.dumps({str(k): int(v) for k, v in df["label_id"].value_counts().sort_index().to_dict().items()}, ensure_ascii=False),
        "UniqueDrugs": len(set(df["drug1_id_std"]) | set(df["drug2_id_std"])),
        "UniqueUnorderedPairs": df["pair_group"].nunique(),
        "NegativeSampling": cfg["negative_sampling"],
        "PreparedCSV": out_path,
    }
    return df, summary, out_path


def make_splits_for_dataset(
    name: str,
    df: pd.DataFrame,
    seeds: List[int],
    n_splits: int,
    out_dir: str,
):
    """
    Generate nested group-aware splits.

    The outer 5-fold split reserves approximately 20% for testing.
    The inner 8-fold split reserves approximately 1/8 of the outer
    training/validation partition for validation, giving an overall
    ratio close to 70%/10%/20%.

    All splitting and all audits use pair_group, so (A, B) and (B, A)
    can never be assigned to different partitions.
    """
    labels = df["label_id"].to_numpy(dtype=np.int64)
    groups = df["pair_group"].to_numpy()
    X = np.zeros(len(df), dtype=np.int8)

    split_dir = os.path.join(out_dir, "splits", name)
    os.makedirs(split_dir, exist_ok=True)

    rows = []
    for seed in seeds:
        outer_splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        for fold, (train_val_idx, test_idx) in enumerate(
            outer_splitter.split(X, labels, groups=groups),
            1,
        ):
            # 1/8 of the outer train_val partition is approximately 10%
            # of the complete dataset. The same pair_group rule is used
            # for the inner train/validation split.
            inner_splitter = StratifiedGroupKFold(
                n_splits=8,
                shuffle=True,
                random_state=seed * 100 + fold,
            )

            inner_X = np.zeros(len(train_val_idx), dtype=np.int8)
            inner_labels = labels[train_val_idx]
            inner_groups = groups[train_val_idx]

            inner_train_pos, val_pos = next(
                inner_splitter.split(
                    inner_X,
                    inner_labels,
                    groups=inner_groups,
                )
            )

            train_idx = np.asarray(train_val_idx[inner_train_pos], dtype=np.int64)
            val_idx = np.asarray(train_val_idx[val_pos], dtype=np.int64)
            test_idx = np.asarray(test_idx, dtype=np.int64)

            # Explicit split-level leakage audit using canonical unordered pairs.
            train_group_set = set(groups[train_idx].tolist())
            val_group_set = set(groups[val_idx].tolist())
            test_group_set = set(groups[test_idx].tolist())

            train_val_overlap = len(train_group_set & val_group_set)
            train_test_overlap = len(train_group_set & test_group_set)
            val_test_overlap = len(val_group_set & test_group_set)

            assert train_val_overlap == 0, (
                f"{name} seed={seed} fold={fold}: "
                f"train/validation pair_group overlap={train_val_overlap}"
            )
            assert train_test_overlap == 0, (
                f"{name} seed={seed} fold={fold}: "
                f"train/test pair_group overlap={train_test_overlap}"
            )
            assert val_test_overlap == 0, (
                f"{name} seed={seed} fold={fold}: "
                f"validation/test pair_group overlap={val_test_overlap}"
            )

            split_path = os.path.join(
                split_dir,
                f"seed_{seed}_fold_{fold}.npz",
            )
            np.savez_compressed(
                split_path,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                seed=np.array([seed], dtype=np.int64),
                fold=np.array([fold], dtype=np.int64),
                group_split=np.array([1], dtype=np.int8),
                pair_group_disjoint=np.array([1], dtype=np.int8),
            )

            rows.append({
                "Dataset": name,
                "Seed": seed,
                "Fold": fold,
                "SplitType": "NestedStratifiedGroupKFold",
                "OuterSplit": f"StratifiedGroupKFold(n_splits={n_splits})",
                "InnerSplit": "StratifiedGroupKFold(n_splits=8)",
                "TrainSize": len(train_idx),
                "ValSize": len(val_idx),
                "TestSize": len(test_idx),
                "TrainUniquePairGroups": len(train_group_set),
                "ValUniquePairGroups": len(val_group_set),
                "TestUniquePairGroups": len(test_group_set),
                "TrainValPairOverlap": train_val_overlap,
                "TrainTestPairOverlap": train_test_overlap,
                "ValTestPairOverlap": val_test_overlap,
                "PairGroupAuditPass": True,
                "TrainClassCounts": safe_count(train_idx, labels),
                "ValClassCounts": safe_count(val_idx, labels),
                "TestClassCounts": safe_count(test_idx, labels),
                "SplitNPZ": split_path,
            })
    return rows


def write_method_setting_template(out_dir: str):
    rows = [
        {
            "Method": "DeepDDI",
            "CodeSource": "Official or official-compatible DeepDDI implementation",
            "InputInformation": "SMILES / molecular structural features",
            "UsesText": "No",
            "UsesNoDDIKG": "No",
            "UsesTrainDDIGraph": "No",
            "OutputLabelSpace": "Dataset-specific: 3/4/4/2 classes",
            "Split": "Read shared split NPZ generated by this script",
            "NegativeSampling": "N/A except PDD-Graph; for PDD read the fixed prepared sample set",
            "HyperparameterBudget": "1 fixed configuration unless otherwise stated",
        },
        {
            "Method": "DrugDAGT",
            "CodeSource": "Official DrugDAGT implementation adapted only for dataset-specific class number and shared split",
            "InputInformation": "SMILES-derived molecular graph, adjacency/distance/Coulomb/atom features",
            "UsesText": "No",
            "UsesNoDDIKG": "No",
            "UsesTrainDDIGraph": "No",
            "OutputLabelSpace": "Dataset-specific: 3/4/4/2 classes",
            "Split": "Read shared split NPZ generated by this script",
            "NegativeSampling": "N/A except PDD-Graph; for PDD read the fixed prepared sample set",
            "HyperparameterBudget": "1 fixed configuration unless otherwise stated",
        },
        {
            "Method": "SumGNN",
            "CodeSource": "Official SumGNN implementation adapted only for dataset-specific relation labels and shared split",
            "InputInformation": "Typed DDI triples plus no-DDI biomedical KG / local KG subgraphs",
            "UsesText": "No",
            "UsesNoDDIKG": "Yes",
            "UsesTrainDDIGraph": "Uses training typed triples only; no validation/test typed DDI edges in KG context",
            "OutputLabelSpace": "Dataset-specific: 3/4/4/2 classes",
            "Split": "Read shared split NPZ generated by this script",
            "NegativeSampling": "N/A for multiclass type classification; PDD fixed binary sample set",
            "HyperparameterBudget": "1 fixed configuration unless otherwise stated",
        },
        {
            "Method": "Node2Vec+MLP",
            "CodeSource": "Official node2vec for embeddings + our MLP downstream classifier",
            "InputInformation": "Training DDI graph topology only; MLP on pair embeddings",
            "UsesText": "No",
            "UsesNoDDIKG": "No",
            "UsesTrainDDIGraph": "Yes; node2vec trained separately on each fold's training graph only",
            "OutputLabelSpace": "Dataset-specific: 3/4/4/2 classes",
            "Split": "Read shared split NPZ generated by this script",
            "NegativeSampling": "N/A except PDD-Graph; for PDD train graph uses positive training edges only",
            "HyperparameterBudget": "1 fixed node2vec config + 1 fixed MLP config",
        },
        {
            "Method": "RA-DDI",
            "CodeSource": "Our method",
            "InputInformation": "SMILES + BioBERT text + no-DDI KG-derived embeddings/features",
            "UsesText": "Yes",
            "UsesNoDDIKG": "Yes",
            "UsesTrainDDIGraph": "No DDI edges in KG embeddings for validation/test leakage prevention",
            "OutputLabelSpace": "Dataset-specific: 3/4/4/2 classes",
            "Split": "Read shared split NPZ generated by this script",
            "NegativeSampling": "N/A except PDD-Graph; for PDD read the fixed prepared sample set",
            "HyperparameterBudget": "1 fixed validated RA-DDI configuration unless otherwise stated",
        },
    ]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "method_input_setting_template.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="D:/study/gnn/DDKG-main/DDKG-main/baseline/split", help="Output directory")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds, e.g. 42,43,44")
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    prepared_dir = os.path.join(args.out_dir, "prepared")

    all_summary = []
    all_manifest = []
    config_records = []

    for name, cfg in DATASETS.items():
        print(f"\n=== Preparing {name} ===")
        df, summary, prepared_path = prepare_dataset(name, cfg, prepared_dir)
        print(f"Prepared rows: {len(df)} | label counts: {df['label_raw'].value_counts().to_dict()}")
        all_summary.append(summary)
        config_records.append({"Dataset": name, **cfg, "prepared_csv": prepared_path})
        split_rows = make_splits_for_dataset(name, df, seeds, args.n_splits, args.out_dir)
        all_manifest.extend(split_rows)
        print(f"Generated {len(split_rows)} split files for {name}.")

    pd.DataFrame(all_summary).to_csv(
        os.path.join(args.out_dir, "dataset_summary.csv"),
        index=False,
    )
    manifest_df = pd.DataFrame(all_manifest)
    manifest_df.to_csv(
        os.path.join(args.out_dir, "split_manifest.csv"),
        index=False,
    )

    audit_columns = [
        "Dataset",
        "Seed",
        "Fold",
        "SplitType",
        "TrainUniquePairGroups",
        "ValUniquePairGroups",
        "TestUniquePairGroups",
        "TrainValPairOverlap",
        "TrainTestPairOverlap",
        "ValTestPairOverlap",
        "PairGroupAuditPass",
        "SplitNPZ",
    ]
    manifest_df[audit_columns].to_csv(
        os.path.join(args.out_dir, "split_pair_leakage_audit.csv"),
        index=False,
    )
    with open(os.path.join(args.out_dir, "split_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": seeds,
                "n_splits": args.n_splits,
                "group_split": True,
                "outer_split": "StratifiedGroupKFold",
                "inner_split": "StratifiedGroupKFold(n_splits=8)",
                "pair_group_rule": "sorted(drug1_id, drug2_id)",
                "datasets": config_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    write_method_setting_template(args.out_dir)

    print("\nDone.")
    print(f"Output directory: {args.out_dir}")
    print("Main files:")
    print("  dataset_summary.csv")
    print("  split_manifest.csv")
    print("  split_pair_leakage_audit.csv")
    print("  method_input_setting_template.csv")
    print("  prepared/<dataset>_prepared.csv")
    print("  splits/<dataset>/seed_<seed>_fold_<fold>.npz")


if __name__ == "__main__":
    main()
