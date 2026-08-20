from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise SystemExit("This benchmark script requires PyTorch. Install torch in the environment first.") from e

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


DRUG1_CANDIDATES = [
    "drug1", "drug_1", "drug_a", "drugA", "DrugA", "Drug1", "Drug1_ID", "drugbank_id_1",
    "head", "h", "source", "src", "left", "left_drug", "d1", "id1",
]
DRUG2_CANDIDATES = [
    "drug2", "drug_2", "drug_b", "drugB", "DrugB", "Drug2", "Drug2_ID", "drugbank_id_2",
    "tail", "t", "target", "dst", "right", "right_drug", "d2", "id2",
]
LABEL_CANDIDATES = [
    "label", "Label", "y", "Y", "class", "Class", "relation", "Relation", "ddi_type", "DDI_type", "type",
]

TRAIN_KEYS = ["train_idx", "train_indices", "idx_train", "train", "train_index", "train_mask"]
VAL_KEYS = ["val_idx", "valid_idx", "validation_idx", "idx_val", "idx_valid", "val", "valid", "validation", "val_mask", "valid_mask"]
TEST_KEYS = ["test_idx", "test_indices", "idx_test", "test", "test_index", "test_mask"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", required=True, help="Dataset name used only for logging/results.")
    p.add_argument("--prepared_csv", required=True, help="Path to split/prepared/<dataset>_prepared.csv")
    p.add_argument("--split_npz", required=True, help="Path to split/splits/<dataset>/seed_<seed>_fold_<fold>.npz")
    p.add_argument("--feature_csv", default="data/drug_tanimoto_PCA50.csv", help="DeepDDI PCA feature csv: drug_id + PC_1...PC_50")
    p.add_argument("--output_dir", required=True, help="Directory for this one seed/fold run")

    p.add_argument("--drug1_col", default=None)
    p.add_argument("--drug2_col", default=None)
    p.add_argument("--label_col", default=None)
    p.add_argument("--label_order", default=None, help="Optional comma-separated label order, e.g. adverse,decrease,increase")

    p.add_argument("--missing_strategy", choices=["error", "zero", "mean"], default="zero",
                   help="How to handle drugs absent from DeepDDI feature_csv. For strict reporting, keep rows and report missing rate; zero is PCA-space mean-like imputation.")

    p.add_argument("--seed", type=int, default=None, help="Training seed. If omitted, parsed from split file name when possible.")
    p.add_argument("--fold", type=int, default=None, help="Fold number. If omitted, parsed from split file name when possible.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10,
                   help="Early-stopping patience. Training stops if the monitored validation metric does not improve for this many epochs.")
    p.add_argument("--monitor_metric", default="f1_macro",
                   help="Validation metric used for early stopping. Must be a key returned by compute_metrics, e.g. f1_macro, accuracy, auprc_macro.")
    p.add_argument("--min_delta", type=float, default=0.0,
                   help="Minimum improvement required to reset early-stopping patience.")
    p.add_argument("--disable_early_stop", action="store_true",
                   help="Disable early stopping and always train for --epochs epochs.")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--num_layers", type=int, default=4, help="Number of hidden Dense-ReLU-Dropout blocks. Original DeepDDI uses 4.")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--class_weight", choices=["none", "balanced"], default="none")
    p.add_argument("--num_workers", type=int, default=0, help="Use 0 on Windows unless you know multiprocessing is safe.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_model", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def detect_col(df: pd.DataFrame, candidates: Sequence[str], explicit: Optional[str], role: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"--{role}_col={explicit!r} not found. Columns={list(df.columns)}")
        return explicit
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in df.columns:
        lc = c.lower()
        if role == "drug1" and re.search(r"(drug|node|entity).*(1|a|left)|^(head|src|source)$", lc):
            return c
        if role == "drug2" and re.search(r"(drug|node|entity).*(2|b|right)|^(tail|dst|target)$", lc):
            return c
        if role == "label" and re.search(r"label|class|relation|type|^y$", lc):
            return c
    raise ValueError(f"Could not auto-detect {role} column. Please pass --{role}_col. Columns={list(df.columns)}")


def read_split(npz_path: str, n_rows: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    z = np.load(npz_path, allow_pickle=True)
    keys = list(z.keys())

    def get_idx(cands: Sequence[str], name: str) -> Tuple[np.ndarray, str]:
        for k in cands:
            if k in z:
                arr = np.asarray(z[k])
                used = k
                break
        else:
            raise ValueError(f"Could not find {name} indices in {npz_path}. Available keys={keys}")
        if arr.dtype == bool:
            if arr.shape[0] != n_rows:
                raise ValueError(f"{used} is a boolean mask of length {arr.shape[0]}, but prepared rows={n_rows}")
            idx = np.flatnonzero(arr)
        else:
            idx = arr.astype(np.int64).reshape(-1)
        if idx.min(initial=0) < 0 or idx.max(initial=-1) >= n_rows:
            raise ValueError(f"{used} has out-of-range index. min={idx.min()}, max={idx.max()}, n_rows={n_rows}")
        return idx, used

    train_idx, train_key = get_idx(TRAIN_KEYS, "train")
    val_idx, val_key = get_idx(VAL_KEYS, "val")
    test_idx, test_key = get_idx(TEST_KEYS, "test")
    return train_idx, val_idx, test_idx, {"train_key": train_key, "val_key": val_key, "test_key": test_key}


def parse_seed_fold(split_npz: str) -> Tuple[Optional[int], Optional[int]]:
    name = Path(split_npz).stem
    m = re.search(r"seed_(\d+)_fold_(\d+)", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def norm_id(x) -> str:
    return str(x).strip().upper()


def load_feature_table(feature_csv: str) -> pd.DataFrame:
    feat = pd.read_csv(feature_csv)
    if feat.shape[1] < 51:
        raise ValueError(f"feature_csv should contain drug id + PC columns. Got shape={feat.shape}: {feature_csv}")
    first = feat.columns[0]
    pc_cols = [c for c in feat.columns if str(c).startswith("PC_")]
    if len(pc_cols) != 50:
        # fallback: use all numeric columns except first
        pc_cols = list(feat.columns[1:51])
        if len(pc_cols) != 50:
            raise ValueError(f"Could not identify 50 PCA feature columns in {feature_csv}")
    feat = feat[[first] + pc_cols].copy()
    feat.columns = ["drug_id"] + [f"PC_{i}" for i in range(1, 51)]
    feat["drug_id"] = feat["drug_id"].map(norm_id)
    feat = feat.drop_duplicates("drug_id", keep="first").set_index("drug_id")
    return feat.astype(np.float32)


def build_pair_features(
    df: pd.DataFrame,
    drug1_col: str,
    drug2_col: str,
    feat: pd.DataFrame,
    missing_strategy: str,
    output_dir: Path,
) -> Tuple[np.ndarray, Dict[str, object]]:
    d1 = df[drug1_col].map(norm_id).to_numpy()
    d2 = df[drug2_col].map(norm_id).to_numpy()
    feat_dim = feat.shape[1]
    assert feat_dim == 50

    feat_index = set(feat.index)
    all_needed = pd.Series(np.concatenate([d1, d2])).drop_duplicates().tolist()
    missing = sorted([x for x in all_needed if x not in feat_index])

    report = {
        "n_unique_drugs_in_prepared": int(len(all_needed)),
        "n_feature_drugs": int(feat.shape[0]),
        "n_missing_drugs": int(len(missing)),
        "missing_rate_unique": float(len(missing) / max(1, len(all_needed))),
        "missing_strategy": missing_strategy,
    }

    if missing:
        pd.DataFrame({"missing_drug_id": missing}).to_csv(output_dir / "missing_feature_drugs.csv", index=False)
        if missing_strategy == "error":
            raise ValueError(
                f"{len(missing)} drugs are missing from {feat.index.name or 'feature table'}; "
                f"examples={missing[:20]}. Either provide a complete DeepDDI PCA feature table or use --missing_strategy zero/mean and report it."
            )

    if missing_strategy == "mean":
        fill_vec = feat.values.mean(axis=0).astype(np.float32)
    else:
        fill_vec = np.zeros(feat_dim, dtype=np.float32)

    def lookup(ids: np.ndarray) -> np.ndarray:
        out = np.empty((len(ids), feat_dim), dtype=np.float32)
        values = feat.to_dict("index")
        # dict of dict is slower but memory-safe enough; for very large files, direct reindex is better:
        tmp = feat.reindex(ids)
        mask = tmp.isna().any(axis=1).to_numpy()
        arr = tmp.fillna(0.0).to_numpy(dtype=np.float32)
        if mask.any():
            arr[mask] = fill_vec
        return arr

    X1 = lookup(d1)
    X2 = lookup(d2)
    X = np.concatenate([X1, X2], axis=1).astype(np.float32)

    row_missing = np.isin(d1, missing) | np.isin(d2, missing)
    report["n_rows_with_any_missing_drug"] = int(row_missing.sum())
    report["row_missing_rate"] = float(row_missing.mean())
    return X, report


def encode_labels(labels: Iterable, label_order: Optional[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    raw = pd.Series(labels).astype(str)
    if label_order:
        classes = [x.strip() for x in label_order.split(",") if x.strip()]
        missing = sorted(set(raw) - set(classes))
        if missing:
            raise ValueError(f"--label_order misses labels present in data: {missing}")
    else:
        uniq = sorted(raw.unique().tolist(), key=lambda x: (not re.fullmatch(r"-?\d+", x), int(x) if re.fullmatch(r"-?\d+", x) else x))
        classes = uniq
    label_to_id = {lab: i for i, lab in enumerate(classes)}
    y = raw.map(label_to_id).to_numpy(dtype=np.int64)
    return y, label_to_id


class DeepDDIMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 1024, num_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            dim = hidden_dim
        layers.append(nn.Linear(dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_loader(X: np.ndarray, y: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    xs = torch.from_numpy(X[idx])
    ys = torch.from_numpy(y[idx])
    ds = TensorDataset(xs, ys)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, idx: np.ndarray, batch_size: int, device: str, num_workers: int) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    loader = make_loader(X, y, idx, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            pred = probs.argmax(axis=1)
            probs_all.append(probs)
            pred_all.append(pred)
    prob = np.concatenate(probs_all, axis=0)
    pred = np.concatenate(pred_all, axis=0)
    true = y[idx]
    metrics = compute_metrics(true, pred, prob)
    return metrics, pred, prob


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    labels = np.arange(prob.shape[1])
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    try:
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        out["balanced_accuracy"] = float("nan")
    for avg in ["macro", "micro", "weighted"]:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        out[f"precision_{avg}"] = float(p)
        out[f"recall_{avg}"] = float(r)
        out[f"f1_{avg}"] = float(f1)
    try:
        if prob.shape[1] == 2:
            out["roc_auc_macro_ovr"] = float(roc_auc_score(y_true, prob[:, 1]))
            out["auprc_macro"] = float(average_precision_score(y_true, prob[:, 1]))
        else:
            y_bin = label_binarize(y_true, classes=labels)
            out["roc_auc_macro_ovr"] = float(roc_auc_score(y_bin, prob, average="macro", multi_class="ovr"))
            out["auprc_macro"] = float(average_precision_score(y_bin, prob, average="macro"))
    except Exception:
        out["roc_auc_macro_ovr"] = float("nan")
        out["auprc_macro"] = float("nan")
    return out


def class_weight_tensor(y_train: np.ndarray, n_classes: int, mode: str, device: str) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(y_train) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    parsed_seed, parsed_fold = parse_seed_fold(args.split_npz)
    seed = args.seed if args.seed is not None else (parsed_seed if parsed_seed is not None else 42)
    fold = args.fold if args.fold is not None else (parsed_fold if parsed_fold is not None else -1)
    set_seed(seed)

    df = pd.read_csv(args.prepared_csv)
    drug1_col = detect_col(df, DRUG1_CANDIDATES, args.drug1_col, "drug1")
    drug2_col = detect_col(df, DRUG2_CANDIDATES, args.drug2_col, "drug2")
    label_col = detect_col(df, LABEL_CANDIDATES, args.label_col, "label")

    train_idx, val_idx, test_idx, split_keys = read_split(args.split_npz, len(df))
    y, label_to_id = encode_labels(df[label_col], args.label_order)
    n_classes = len(label_to_id)
    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes, got label_to_id={label_to_id}")

    feat = load_feature_table(args.feature_csv)
    X, feature_report = build_pair_features(df, drug1_col, drug2_col, feat, args.missing_strategy, outdir)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = "cpu"

    model = DeepDDIMLP(input_dim=X.shape[1], n_classes=n_classes, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout).to(device)
    weights = class_weight_tensor(y[train_idx], n_classes, args.class_weight, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = make_loader(X, y, train_idx, args.batch_size, shuffle=True, num_workers=args.num_workers)

    best_val = -float("inf")
    best_epoch = 0
    best_state = None
    bad = 0
    stopped_epoch = 0
    early_stop_triggered = False
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            bs = xb.shape[0]
            total_loss += float(loss.item()) * bs
            n_seen += bs

        val_metrics, _, _ = evaluate(model, X, y, val_idx, args.batch_size, device, args.num_workers)
        train_loss = total_loss / max(1, n_seen)
        if args.monitor_metric not in val_metrics:
            raise ValueError(
                f"--monitor_metric={args.monitor_metric!r} is not available. "
                f"Available validation metrics: {sorted(val_metrics.keys())}"
            )
        monitor = float(val_metrics[args.monitor_metric])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "monitor_metric": args.monitor_metric,
            "monitor_value": monitor,
            "best_monitor_value_so_far": max(best_val, monitor),
            "bad_epochs_before_update": bad,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_{args.monitor_metric}={monitor:.6f} "
            f"val_f1_macro={val_metrics.get('f1_macro', float('nan')):.6f} "
            f"val_acc={val_metrics.get('accuracy', float('nan')):.6f}"
        )

        improved = monitor > (best_val + args.min_delta)
        if improved:
            best_val = monitor
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if (not args.disable_early_stop) and args.patience > 0 and bad >= args.patience:
                stopped_epoch = epoch
                early_stop_triggered = True
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best_epoch={best_epoch}, best_val_{args.monitor_metric}={best_val:.6f}, "
                    f"patience={args.patience}, min_delta={args.min_delta}"
                )
                break

    epochs_run = len(history)
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, y_pred, prob = evaluate(model, X, y, test_idx, args.batch_size, device, args.num_workers)
    id_to_label = {v: k for k, v in label_to_id.items()}

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
    with open(outdir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, ensure_ascii=False, indent=2)
    with open(outdir / "feature_report.json", "w", encoding="utf-8") as f:
        json.dump(feature_report, f, ensure_ascii=False, indent=2)

    pred_df = df.iloc[test_idx][[drug1_col, drug2_col, label_col]].copy()
    pred_df.insert(0, "row_id", test_idx)
    pred_df["y_true_id"] = y[test_idx]
    pred_df["y_true"] = [id_to_label[int(i)] for i in y[test_idx]]
    pred_df["y_pred_id"] = y_pred
    pred_df["y_pred"] = [id_to_label[int(i)] for i in y_pred]
    for i in range(n_classes):
        pred_df[f"prob_{id_to_label[i]}"] = prob[:, i]
    pred_df.to_csv(outdir / "predictions_test.csv", index=False)

    if args.save_model:
        torch.save({"model_state": model.state_dict(), "args": vars(args), "label_to_id": label_to_id}, outdir / "model.pt")

    result_row = {
        "method": "DeepDDI_shared",
        "dataset": args.dataset,
        "seed": seed,
        "fold": fold,
        "prepared_csv": str(Path(args.prepared_csv).resolve()),
        "split_npz": str(Path(args.split_npz).resolve()),
        "drug1_col": drug1_col,
        "drug2_col": drug2_col,
        "label_col": label_col,
        "n_rows_prepared": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_classes": int(n_classes),
        "input_dim": int(X.shape[1]),
        "feature_csv": str(Path(args.feature_csv).resolve()),
        "missing_strategy": args.missing_strategy,
        "n_missing_drugs": feature_report["n_missing_drugs"],
        "row_missing_rate": feature_report["row_missing_rate"],
        "architecture": f"MLP({X.shape[1]}-{args.hidden_dim}x{args.num_layers}-{n_classes}), ReLU, dropout={args.dropout}",
        "optimizer": "Adam",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "epochs_run": epochs_run,
        "early_stop_triggered": early_stop_triggered,
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
        "monitor_metric": args.monitor_metric,
        "best_val_monitor": best_val,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "disable_early_stop": args.disable_early_stop,
        "class_weight": args.class_weight,
        "elapsed_sec": time.time() - start,
        **test_metrics,
    }
    pd.DataFrame([result_row]).to_csv(outdir / "metrics.csv", index=False)
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(result_row, f, ensure_ascii=False, indent=2)

    print("\n=== TEST METRICS ===")
    for k in ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_macro_ovr", "auprc_macro"]:
        print(f"{k}: {test_metrics.get(k, float('nan')):.6f}")
    print(f"Saved to: {outdir}")


if __name__ == "__main__":
    main()
