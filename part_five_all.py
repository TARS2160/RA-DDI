import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import os
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.metrics import precision_score, recall_score
# ========== 通用配置 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

EPOCHS = 50
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_SPLITS = 5

# ========== 数据加载 ==========
class DDIDataset(Dataset):
    def __init__(self, df, drug_ids, embeddings, label_map):
        self.drug2idx = {did: i for i, did in enumerate(drug_ids)}
        self.embeddings = embeddings
        self.pairs, self.labels = [], []
        for _, row in df.iterrows():
            d1, d2, lbl = row["first drug id"], row["second drug id"], row["label"]
            if d1 in self.drug2idx and d2 in self.drug2idx:
                self.pairs.append((self.drug2idx[d1], self.drug2idx[d2]))
                self.labels.append(label_map[lbl])
        print(f"有效样本数: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        d1, d2 = self.pairs[idx]
        label = self.labels[idx]
        return torch.tensor(d1), torch.tensor(d2), torch.tensor(label)

# ========== 对ogbl-ddi数据库进行处理  ==========
def process_ogbl_ddi_dataset(path, min_samples=1000):
    df = pd.read_csv(path)
    print(f"原始DDI数据大小: {len(df)}")

    label_counts = df["label"].value_counts()
    valid_labels = label_counts[label_counts >= min_samples].index.tolist()
    df_filtered = df[df["label"].isin(valid_labels)].copy()

    print(f"过滤后数据大小: {len(df_filtered)}")
    print("过滤后类别分布:")
    print(df_filtered["label"].value_counts())

    labels_unique = df_filtered["label"].unique().tolist()
    label_map = {lbl: i for i, lbl in enumerate(labels_unique)}
    print(f"类别数量: {len(label_map)}")

    return df_filtered, label_map


# ========== 对DDInter数据库进行改进处理 ==========
def process_ddinter_dataset(path: str, min_samples: int = 100):
    """
    改进版 DDInter 数据处理函数
    保留原始接口: 返回 df_filtered, label_map
    """
    print("=== 加载 DDInter 数据 ===")
    df = pd.read_csv(path)
    print(f"原始数据大小: {len(df)}")
    print(f"列名: {df.columns.tolist()}")

    # ===============================
    # 1. 标签标准化与映射
    # ===============================
    label_map_default = {"Moderate": 0, "Minor": 1, "Major": 2}

    if "label" not in df.columns:
        raise ValueError("❌ 缺少 label 列，无法处理 DDInter 数据。")

    if df["label"].dtype == "object":
        unique_labels = df["label"].unique().tolist()
        print(f"检测到文本标签: {unique_labels}")

        # 如果标签符合预期，则直接映射
        if set(unique_labels).issubset(label_map_default.keys()):
            df["label"] = df["label"].map(label_map_default)
            label_map = label_map_default
            print("✅ 使用默认标签映射:", label_map)
        else:
            print("⚠️ 标签不符合预期，使用自动编码器...")
            le = LabelEncoder()
            df["label"] = le.fit_transform(df["label"])
            label_map = dict(zip(le.classes_, range(len(le.classes_))))
            print(f"✅ 自动编码标签映射: {label_map}")
    else:
        label_map = {lbl: i for i, lbl in enumerate(sorted(df["label"].unique()))}
        print(f"检测到数值标签: {label_map}")

    # ===============================
    # 2. 过滤低频标签
    # ===============================
    label_counts = df["label"].value_counts()
    print(f"类别分布:\n{label_counts}")

    valid_labels = [lbl for lbl, c in label_counts.items() if c >= min_samples]
    if len(valid_labels) < len(label_counts):
        print(f"过滤掉以下低频类别: {[lbl for lbl in label_counts.index if lbl not in valid_labels]}")

    df_filtered = df[df["label"].isin(valid_labels)].copy()
    print(f"过滤后样本数: {len(df_filtered)}")
    print(f"过滤后类别分布:\n{df_filtered['label'].value_counts()}")

    # ===============================
    # 3. 重建标签映射（保持连续性）
    # ===============================
    labels_unique = sorted(df_filtered["label"].unique().tolist())
    label_map = {lbl: i for i, lbl in enumerate(labels_unique)}
    df_filtered["label"] = df_filtered["label"].map(label_map)
    print(f"✅ 最终标签映射: {label_map}")

    return df_filtered, label_map



# ========== 对PDD Graph数据库进行改进处理 ==========
def process_pdd_graph_dataset(path: str, min_samples: int = 10):
    """
    改进版 PDD Graph 数据处理函数
    保留原始接口: 返回 df_filtered, label_map
    """
    print("=== 加载 PDD Graph 数据 ===")
    df = pd.read_csv(path)
    print(f"原始数据大小: {len(df)}")
    print(f"列名: {df.columns.tolist()}")

    # ===============================
    # 1. 检查列名一致性
    # ===============================
    expected_cols = ["first drug id", "second drug id"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"❌ 缺少必须的列: {col}")

    # ===============================
    # 2. 生成二分类标签
    # ===============================
    print("🔄 生成正负样本标签...")
    df["label"] = 1  # 所有现有为正例

    positives = df[["first drug id", "second drug id"]].copy()
    positives["label"] = 1

    # 负采样（1:1比例）
    drugs = list(pd.concat([df["first drug id"], df["second drug id"]]).unique())
    pos_set = set(zip(positives["first drug id"], positives["second drug id"]))

    num_negatives = len(positives)
    negatives = set()
    while len(negatives) < num_negatives:
        d1, d2 = random.sample(drugs, 2)
        if (d1, d2) not in pos_set and (d2, d1) not in pos_set:
            negatives.add((d1, d2))

    negatives = pd.DataFrame(list(negatives), columns=["first drug id", "second drug id"])
    negatives["label"] = 0

    df_combined = pd.concat([positives, negatives], ignore_index=True)
    print(f"✅ 正样本: {len(positives)}, 负样本: {len(negatives)}")
    print(f"合并后样本数: {len(df_combined)}")

    # ===============================
    # 3. 标签过滤（PDD Graph 一般无需）
    # ===============================
    label_counts = df_combined["label"].value_counts()
    print(f"类别分布:\n{label_counts}")

    df_filtered = df_combined.copy()
    valid_labels = [0, 1]
    label_map = {lbl: i for i, lbl in enumerate(valid_labels)}
    df_filtered["label"] = df_filtered["label"].map(label_map)

    print(f"✅ 标签映射: {label_map}")
    print(f"最终样本数: {len(df_filtered)}")

    return df_filtered, label_map


# === 0 OurDDI ===
class RelationAwareDDIPredictor(nn.Module):
    def __init__(self, emb_dim, num_classes, feature_dim=4, hidden_dim=512):
        super().__init__()

        self.drug1_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.drug2_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.interaction_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2 + 32,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 32, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, d1_emb, d2_emb, features=None):
        """
        支持两种调用方式:
        - model(d1_emb, d2_emb): 自动生成pair特征
        - model(d1_emb, d2_emb, features): 使用外部提供特征
        """

        # 如果未提供features，则自动生成
        if features is None or features.ndim != 2 or features.shape[1] != 4:
            features = torch.stack([
                F.cosine_similarity(d1_emb, d2_emb, dim=1),
                torch.norm(d1_emb - d2_emb, dim=1),
                (d1_emb * d2_emb).sum(dim=1),
                torch.abs(d1_emb - d2_emb).mean(dim=1)
            ], dim=1)

        # 药物特征投影
        d1_feat = self.drug1_proj(d1_emb)
        d2_feat = self.drug2_proj(d2_emb)

        # 交互特征
        interaction = torch.cat([d1_feat, d2_feat], dim=1)
        interaction_feat = self.interaction_net(interaction)

        # 相似度特征
        feature_emb = self.feature_net(features)

        # 合并特征
        combined = torch.cat([interaction_feat, feature_emb], dim=1)

        # 注意力机制
        combined = combined.unsqueeze(1)
        attended, _ = self.attention(combined, combined, combined)
        attended = attended.squeeze(1)

        return self.classifier(attended)

# === 1️⃣ MLP-DDI（DeepDDI / Node2Vec+MLP） ===
class MLPDDI(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, d1_emb, d2_emb):
        x = torch.cat([d1_emb, d2_emb], dim=1)
        return self.fc(x)


# === 2️⃣ GCN-DDI ===
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj=None):
        # 简化图卷积：此处不使用真实邻接矩阵，仅保留结构变换
        out = self.linear(x)
        out = self.bn(out)
        return F.relu(out)


class GCN_DDI(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.gcn1 = GCNLayer(emb_dim, 512)
        self.gcn2 = GCNLayer(512, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, d1_emb, d2_emb):
        h1 = self.gcn2(self.gcn1(d1_emb))
        h2 = self.gcn2(self.gcn1(d2_emb))
        out = torch.cat([h1, h2], dim=1)
        out = self.dropout(out)
        return self.fc(out)


# === 3️⃣ GAT-DDI  ===
class GAT_DDI(nn.Module):
    def __init__(self, emb_dim, num_classes, heads=8):
        super().__init__()
        self.heads = heads
        self.att1 = nn.MultiheadAttention(emb_dim, num_heads=heads, batch_first=True, dropout=0.2)
        self.att2 = nn.MultiheadAttention(emb_dim, num_heads=heads, batch_first=True, dropout=0.2)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        # 门控融合层 (平衡差值与乘积特征)
        self.gate_fc = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Sigmoid()
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, d1_emb, d2_emb):
        x = torch.stack([d1_emb, d2_emb], dim=1)  # [B, 2, D]

        attn_out1, _ = self.att1(x, x, x)
        x = self.norm1(attn_out1 + x)

        attn_out2, _ = self.att2(x, x, x)
        x = self.norm2(attn_out2 + x)  # [B, 2, D]

        # 融合层
        d1_out, d2_out = x[:, 0, :], x[:, 1, :]
        diff = torch.abs(d1_out - d2_out)
        prod = d1_out * d2_out
        gate = self.gate_fc(torch.cat([diff, prod], dim=1))
        fused = gate * diff + (1 - gate) * prod  # 门控融合

        x_cat = torch.cat([fused, d1_out + d2_out], dim=1)
        return self.fc(x_cat)

# === 4️⃣ SkipGNN-DDI ===
class SkipGNN_DDI(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        self.fc_skip = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.fc_out = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, d1_emb, d2_emb):
        h1 = self.fc1(d1_emb)
        h2 = self.fc_skip(d2_emb)
        out = torch.cat([h1, h2], dim=1)
        fused = self.fusion(out)
        return self.fc_out(fused)

# === 5️⃣ KGNN-DDI ===
class KGNN_DDI(nn.Module):
    def __init__(self, emb_dim, num_classes, rel_dim=256):
        super().__init__()
        self.entity_transform = nn.Sequential(
            nn.Linear(emb_dim, rel_dim),
            nn.BatchNorm1d(rel_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(rel_dim, rel_dim),
            nn.ReLU()
        )

        self.rel_transform = nn.Sequential(
            nn.Linear(rel_dim, rel_dim),
            nn.LayerNorm(rel_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # 门控关系融合
        self.gate_fc = nn.Sequential(
            nn.Linear(rel_dim * 2, rel_dim),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(rel_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, d1_emb, d2_emb):
        d1_r = self.entity_transform(d1_emb)
        d2_r = self.entity_transform(d2_emb)

        # 显式关系建模
        diff = torch.abs(d1_r - d2_r)
        prod = d1_r * d2_r
        rel_feat = self.rel_transform(diff + prod)

        # 门控融合
        gate = self.gate_fc(torch.cat([diff, prod], dim=1))
        fused_rel = gate * rel_feat + (1 - gate) * diff

        # 残差 + 拼接
        x = torch.cat([fused_rel, d1_r + d2_r], dim=1)
        return self.fc(x)


# ========== 评估函数 ==========
def evaluate(y_true, y_pred, y_prob, num_classes):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    # --- 计算多分类 ROC / PR ---
    try:
        if num_classes == 2:
            auc_roc = roc_auc_score(y_true, y_prob)
            auc_pr = average_precision_score(y_true, y_prob)
        else:
            # 将标签转成 one-hot
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
            # y_prob 应该是每类的概率分布
            y_prob = np.array(y_prob)
            if y_prob.ndim == 1:
                y_prob = np.expand_dims(y_prob, axis=1)
            auc_roc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
            auc_pr = average_precision_score(y_true_bin, y_prob, average='macro')
    except Exception as e:
        print(f"[Warn] AUC计算失败: {e}")
    return {"Accuracy": acc, "Macro-F1": f1, "AUC-ROC": auc_roc, "AUC-PR": auc_pr}


def train_baseline(model_name, dataset, embeddings, num_classes):
    """
    针对每个模型执行五折交叉验证训练 + 指标评估
    """
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    results = []
    emb_dim = embeddings.shape[1]

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(dataset)), dataset.labels), 1):
        print(f"\n===== {model_name} | Fold {fold}/{N_SPLITS} =====")

        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=1024)

        # 初始化模型
        if model_name == "DeepDDI":
            model = MLPDDI(emb_dim, num_classes)
        elif model_name == "Node2Vec+MLP":
            model = MLPDDI(emb_dim, num_classes)
        elif model_name == "GCN-DDI":
            model = GCN_DDI(emb_dim, num_classes)
        elif model_name == "GAT-DDI":
            model = GAT_DDI(emb_dim, num_classes)
        elif model_name == "SkipGNN":
            model = SkipGNN_DDI(emb_dim, num_classes)
        elif model_name == "KGNN-DDI":
            model = KGNN_DDI(emb_dim, num_classes)
        elif model_name == "RA-DDI":
            model = RelationAwareDDIPredictor(emb_dim, num_classes)
        else:
            raise ValueError(f"未知模型：{model_name}")

        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        # ---------- Epoch 训练 ----------
        epoch_bar = tqdm(range(EPOCHS), desc=f"{model_name} | Fold {fold}", ncols=100)
        for epoch in epoch_bar:
            model.train()
            total_loss = 0.0
            for d1, d2, label in train_loader:
                d1, d2, label = d1.to(device), d2.to(device), label.to(device)
                d1_emb, d2_emb = embeddings[d1], embeddings[d2]
                if isinstance(model, RelationAwareDDIPredictor):
                    out = model(d1_emb, d2_emb)  # 新RA-DDI已支持不传features
                else:
                    out = model(d1_emb, d2_emb)
                loss = criterion(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            epoch_bar.set_postfix({"Loss": f"{avg_loss:.4f}"})

        # ---------- 测试评估 ----------
        model.eval()
        preds, probs, trues = [], [], []
        with torch.no_grad():
            for d1, d2, label in test_loader:
                d1, d2 = d1.to(device), d2.to(device)
                d1_emb, d2_emb = embeddings[d1], embeddings[d2]
                logits = model(d1_emb, d2_emb)
                p = F.softmax(logits, dim=1)
                preds.extend(p.argmax(1).cpu().tolist())
                probs.extend(p.cpu().tolist())
                trues.extend(label.tolist())

        # ---------- 计算详细指标 ----------
        acc = accuracy_score(trues, preds)
        macro_p = precision_score(trues, preds, average='macro', zero_division=0)
        macro_r = recall_score(trues, preds, average='macro', zero_division=0)
        macro_f1 = f1_score(trues, preds, average='macro', zero_division=0)

        try:
            if num_classes == 2:
                y_prob = np.array([p[1] for p in probs])
                auc_roc = roc_auc_score(trues, y_prob)
                auc_pr = average_precision_score(trues, y_prob)
            else:
                y_true_bin = label_binarize(trues, classes=list(range(num_classes)))
                auc_roc = roc_auc_score(y_true_bin, np.array(probs), average='macro', multi_class='ovr')
                auc_pr = average_precision_score(y_true_bin, np.array(probs), average='macro')
        except Exception as e:
            print(f"[Warn] AUC计算失败: {e}")
            auc_roc, auc_pr = np.nan, np.nan

        fold_result = {
            "Accuracy": acc,
            "Precision": macro_p,
            "Recall": macro_r,
            "Macro-F1": macro_f1,
            "AUC-ROC": auc_roc,
            "AUC-PR": auc_pr
        }
        results.append(fold_result)
        print(f"Fold {fold} | "
              f"Acc={acc:.4f} | P={macro_p:.4f} | R={macro_r:.4f} | F1={macro_f1:.4f} | "
              f"AUC={auc_roc:.4f}")

    # ---------- 平均五折结果 ----------
    return {k: np.nanmean([r[k] for r in results]) for k in results[0]}

# ========== 主函数：多数据集自动跑 ==========
def run_all_datasets():
    DATASETS = {
        "DDInter": "data/ddinter/test3.csv",
        "PDD-Graph": "data/PDD_graph/test2.csv"
    }
    # "ogbl-DDI": "data/ogb_ddi/ogbl_ddi/mapping/test1.csv"
    all_models = ["RA-DDI", "DeepDDI", "Node2Vec+MLP", "GCN-DDI", "GAT-DDI", "SkipGNN", "KGNN-DDI"]
    summary = []

    # 加载药物全局表示
    emb_data = torch.load("models/model/drug_global_embeddings.pt")
    embeddings = emb_data["global_embeddings"].to(device)
    drug_ids = emb_data["drug_ids"]

    # results 文件夹
    os.makedirs("results", exist_ok=True)

    # 遍历数据集
    for dname, path in DATASETS.items():
        print(f"\n==== 开始数据集 {dname}, path={path} ====")
        # df = pd.read_csv(path)
        # label_map = {v: i for i, v in enumerate(sorted(df["label"].unique()))}
        if dname == "ogbl-DDI":
            df, label_map = process_ogbl_ddi_dataset(path, min_samples=1000)
        elif dname == "DDInter":
            df, label_map = process_ddinter_dataset(path, min_samples=100)
        else:
            df, label_map = process_pdd_graph_dataset(path, min_samples=10)
        dataset = DDIDataset(df, drug_ids, embeddings.cpu(), label_map)
        # dataset.labels = [label_map[l] for l in df["label"]]
        num_classes = len(label_map)

        print(f"类别数量: {num_classes}, label_map: {label_map}")
        print(f"样本数: {len(dataset)}")

        dataset_results = []  # 当前数据集结果

        # tqdm 进度条
        for model_name in tqdm(all_models, desc=f"训练中 ({dname})", ncols=100):
            print(f"\n>>> 开始运行模型: {model_name}")
            result = train_baseline(model_name, dataset, embeddings, num_classes)

            # 打印结果
            print(f"✅ 模型 {model_name} 在 {dname} 上结果:")
            print(f"   Accuracy: {result['Accuracy']:.4f} | Macro-F1: {result['Macro-F1']:.4f} | "
                  f"AUC-ROC: {result['AUC-ROC']:.4f} | AUC-PR: {result['AUC-PR']:.4f}")

            result_row = {
                "Dataset": dname,
                "Model": model_name,
                **result
            }

            summary.append(result_row)
            dataset_results.append(result_row)

        # 当前数据集结果保存
        df_dataset = pd.DataFrame(dataset_results)
        csv_path = f"results/compare/{dname}_results.csv"
        df_dataset.to_csv(csv_path, index=False)
        print(f"📁 {dname} 数据集结果已保存至: {csv_path}")

    # 汇总所有结果
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv("results/compare/baseline_results_summary.csv", index=False)
    print("\n✅ 全部实验完成，汇总结果已保存至 results/compare/baseline_results_summary.csv")


if __name__ == "__main__":
    run_all_datasets()
