# part_four_ddi_improved.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score, roc_curve, auc, \
    precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter, defaultdict
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
import os
import random
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")


# =======================
# 改进的数据集类
# =======================
class EnhancedDDIDataset(Dataset):
    def __init__(self, df, drug_ids, embeddings, label_map):
        self.drug2idx = {did: i for i, did in enumerate(drug_ids)}
        self.embeddings = embeddings

        self.pairs = []
        self.labels = []
        self.label_indices = defaultdict(list)

        # 统计匹配情况
        ddi_drugs = set(df['first drug id'].tolist() + df['second drug id'].tolist())
        matched_drugs = ddi_drugs.intersection(set(drug_ids))
        print(f"DDI数据中的药物数量: {len(ddi_drugs)}")
        print(f"匹配的药物数量: {len(matched_drugs)}")
        print(f"匹配率: {len(matched_drugs) / len(ddi_drugs):.2%}")

        skipped = 0
        for idx, row in df.iterrows():
            d1, d2, lbl = row["first drug id"], row["second drug id"], row["label"]
            if lbl not in label_map:
                skipped += 1
                continue
            if d1 in self.drug2idx and d2 in self.drug2idx:
                label_idx = label_map[lbl]
                self.pairs.append((self.drug2idx[d1], self.drug2idx[d2]))
                self.labels.append(label_idx)
                self.label_indices[label_idx].append(len(self.pairs) - 1)
            else:
                skipped += 1

        print(f"总样本数: {len(df)}, 有效样本数: {len(self.pairs)}, 跳过样本数: {skipped}")
        print("📊 类别分布:", Counter(self.labels))

        # 预计算相似度特征
        self.similarity_features = self._compute_similarity_features()

    def _compute_similarity_features(self):
        """预计算相似度特征"""
        print("计算相似度特征...")
        norms = torch.norm(self.embeddings, dim=1, keepdim=True)
        normalized_emb = self.embeddings / norms

        features = []
        for idx1, idx2 in self.pairs:
            emb1 = self.embeddings[idx1]
            emb2 = self.embeddings[idx2]
            norm_emb1 = normalized_emb[idx1]
            norm_emb2 = normalized_emb[idx2]

            # 多种相似度度量
            cosine_sim = torch.dot(norm_emb1, norm_emb2).item()
            euclidean_dist = torch.dist(emb1, emb2).item()
            manhattan_dist = torch.sum(torch.abs(emb1 - emb2)).item()
            dot_product = torch.dot(emb1, emb2).item()

            features.append([cosine_sim, euclidean_dist, manhattan_dist, dot_product])

        return torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = torch.tensor(self.pairs[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        features = self.similarity_features[idx]
        return pair, label, features


# =======================
# 关系感知的DDI预测模型
# =======================
class RelationAwareDDIPredictor(nn.Module):
    def __init__(self, emb_dim, num_classes, feature_dim=4, hidden_dim=512):
        super().__init__()

        # 药物投影层
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

        # 交互特征提取
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

        # 相似度特征处理
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2 + 32,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )

        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 32, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, d1_emb, d2_emb, features):
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


# =======================
# 改进的损失函数
# =======================
class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        if self.smoothing > 0:
            # 标签平滑
            num_classes = inputs.size(-1)
            targets = F.one_hot(targets, num_classes).float()
            targets = (1 - self.smoothing) * targets + self.smoothing / num_classes
            log_probs = F.log_softmax(inputs, dim=-1)
            loss = - (targets * log_probs).sum(dim=-1)
        else:
            loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # Focal loss
        pt = torch.exp(-loss)
        focal_loss = ((1 - pt) ** self.gamma) * loss

        return focal_loss.mean()


# =======================
# 综合评估函数
# =======================
def comprehensive_evaluation(true_labels, predictions, probabilities, num_classes):
    """全面的评估指标计算（兼容二分类和多分类）"""
    metrics = {}

    # 基础分类指标
    metrics['accuracy'] = accuracy_score(true_labels, predictions)
    metrics['macro_precision'] = precision_score(true_labels, predictions, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(true_labels, predictions, average='macro', zero_division=0)
    metrics['macro_f1'] = f1_score(true_labels, predictions, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(true_labels, predictions, average='weighted', zero_division=0)

    # 每个类别的精确率召回率
    per_class_precision = precision_score(true_labels, predictions, average=None, zero_division=0)
    per_class_recall = recall_score(true_labels, predictions, average=None, zero_division=0)
    metrics['per_class_precision'] = per_class_precision
    metrics['per_class_recall'] = per_class_recall

    # AUC指标
    try:
        if probabilities is not None and len(probabilities) > 0:
            # 转成 numpy，避免 list 出错
            probabilities = np.array(probabilities)

            if num_classes == 2:
                # 二分类：直接用正类概率
                if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                    y_score = probabilities[:, 1]  # 取正类概率
                else:
                    # 已经是一维概率，或者需要重塑
                    if probabilities.ndim > 1 and probabilities.shape[1] == 1:
                        y_score = probabilities.ravel()
                    else:
                        y_score = probabilities
                metrics['macro_auc_roc'] = roc_auc_score(true_labels, y_score)
                metrics['macro_auc_pr'] = average_precision_score(true_labels, y_score)
            else:
                # 多分类：用 one-vs-rest
                y_true_bin = label_binarize(true_labels, classes=range(num_classes))
                metrics['macro_auc_roc'] = roc_auc_score(y_true_bin, probabilities, average='macro', multi_class='ovr')
                metrics['macro_auc_pr'] = average_precision_score(y_true_bin, probabilities, average='macro')
    except Exception as e:
        print(f"AUC计算警告: {e}")
        metrics['macro_auc_roc'] = 0.0
        metrics['macro_auc_pr'] = 0.0

    return metrics


def plot_confusion_matrix(true_labels, predictions, class_names, fold=None, save_dir=None, ddi_source=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    title = f'Confusion Matrix'
    if fold:
        title += f' - Fold {fold}'
    if ddi_source:
        title += f' - {ddi_source}'

    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    fold_str = f"_fold{fold}" if fold else ""
    source_str = ddi_source.replace(' ', '_') if ddi_source else ""

    filename = f"confusion_matrix{fold_str}_{source_str}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return cm


# =======================
# 曲线绘制函数
# =======================
def plot_roc_curve(y_true, y_score, n_classes, class_names, save_path, title_suffix="", multi_class=True):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 8))

    if n_classes == 2:
        # 二分类情况
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1] if y_score.ndim == 2 and y_score.shape[1] > 1 else y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        # 多分类情况
        # One-vs-Rest策略
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # 每个类别的曲线
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=1.5,
                     label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

        # 宏平均曲线
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        roc_auc_macro = auc(all_fpr, mean_tpr)

        plt.plot(all_fpr, mean_tpr, color='darkred', linestyle='--', lw=3,
                 label=f'Macro-average (AUC = {roc_auc_macro:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    title = 'Receiver Operating Characteristic (ROC) Curve'
    if title_suffix:
        title += f" - {title_suffix}"
    plt.title(title)

    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return roc_auc_macro if n_classes > 2 else roc_auc


def plot_pr_curve(y_true, y_score, n_classes, class_names, save_path, title_suffix=""):
    """绘制PR曲线"""
    plt.figure(figsize=(10, 8))

    if n_classes == 2:
        # 二分类情况
        precision, recall, _ = precision_recall_curve(y_true, y_score[:, 1] if y_score.ndim == 2 and y_score.shape[
            1] > 1 else y_score)
        aupr = auc(recall, precision)

        # 计算PR-AUC
        aupr_score = average_precision_score(y_true, y_score) if y_score.ndim == 1 else average_precision_score(y_true,
                                                                                                                y_score[
                                                                                                                :, 1])
        plt.plot(recall, precision, color='darkgreen', lw=2,
                 label=f'PR curve (AUPR = {aupr:.3f})')
    else:
        # 多分类情况
        # One-vs-Rest策略
        precision = dict()
        recall = dict()
        aupr = dict()
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # 每个类别的曲线
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
            aupr[i] = auc(recall[i], precision[i])
            plt.plot(recall[i], precision[i], lw=1.5,
                     label=f'{class_names[i]} (AUPR = {aupr[i]:.3f})')

        # 宏平均曲线
        all_recall = np.unique(np.concatenate([recall[i] for i in range(n_classes)]))
        mean_precision = np.zeros_like(all_recall)
        for i in range(n_classes):
            mean_precision += np.interp(all_recall, recall[i], precision[i])
        mean_precision /= n_classes
        aupr_macro = auc(all_recall, mean_precision)

        plt.plot(all_recall, mean_precision, color='darkblue', linestyle='--', lw=3,
                 label=f'Macro-average (AUPR = {aupr_macro:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    title = 'Precision-Recall (PR) Curve'
    if title_suffix:
        title += f" - {title_suffix}"
    plt.title(title)

    plt.legend(loc="upper right")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return aupr_macro if n_classes > 2 else aupr


def plot_fold_curves(true_labels, predictions, probabilities, num_classes, class_names, save_dir, fold,
                     ddi_source=None):
    """绘制单个折的ROC和PR曲线"""
    # 创建文件夹
    os.makedirs(save_dir, exist_ok=True)

    true_labels = np.array(true_labels)
    probabilities = np.array(probabilities)

    # 标题后缀
    title_suffix = f"Fold {fold}"
    if ddi_source:
        title_suffix += f" - {ddi_source}"

    # ROC曲线
    roc_path = os.path.join(save_dir, f"roc_curve_fold{fold}_{ddi_source.replace(' ', '_') if ddi_source else ''}.png")
    plot_roc_curve(true_labels, probabilities, num_classes, class_names, roc_path, title_suffix)

    # PR曲线
    pr_path = os.path.join(save_dir, f"pr_curve_fold{fold}_{ddi_source.replace(' ', '_') if ddi_source else ''}.png")
    plot_pr_curve(true_labels, probabilities, num_classes, class_names, pr_path, title_suffix)


def plot_all_curves(all_fold_results, num_classes, class_names, save_dir, ddi_source=None):
    """绘制所有折合并后的ROC和PR曲线"""
    # 收集所有折的预测结果
    all_true = []
    all_probs = []

    for fold_result in all_fold_results:
        all_true.extend(fold_result['true_labels'])
        all_probs.extend(fold_result['probabilities'])

    all_true = np.array(all_true)
    all_probs = np.array(all_probs)

    # 标题后缀
    title_suffix = "Overall Results"
    if ddi_source:
        title_suffix += f" - {ddi_source}"

    # ROC曲线
    roc_path = os.path.join(save_dir, f"roc_curve_overall_{ddi_source.replace(' ', '_') if ddi_source else ''}.png")
    roc_auc = plot_roc_curve(all_true, all_probs, num_classes, class_names, roc_path, title_suffix)

    # PR曲线
    pr_path = os.path.join(save_dir, f"pr_curve_overall_{ddi_source.replace(' ', '_') if ddi_source else ''}.png")
    pr_auc = plot_pr_curve(all_true, all_probs, num_classes, class_names, pr_path, title_suffix)

    print(f"整体 ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    return roc_auc, pr_auc


# =======================
# 5折交叉验证训练函数
# =======================
def cross_validation_training(dataset, embeddings, num_classes, ddi_source, n_splits=5, epochs=50):
    """5折交叉验证训练"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    all_fold_results = []

    # 为当前数据库和折创建结果目录
    confusion_dir = f"results/{ddi_source.replace(' ', '_')}/confusion_matrices"
    curve_dir = f"results/{ddi_source.replace(' ', '_')}/curves"
    os.makedirs(confusion_dir, exist_ok=True)
    os.makedirs(curve_dir, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), dataset.labels), 1):
        print(f"\n{'=' * 60}")
        print(f"🎯 开始第 {fold}/{n_splits} 折训练 - {ddi_source}")
        print(f"{'=' * 60}")

        # 分割训练集和验证集
        train_sub_idx, val_idx = train_test_split(
            train_idx, test_size=0.125, random_state=42,
            stratify=[dataset.labels[i] for i in train_idx]
        )

        # 创建数据加载器
        train_sampler = torch.utils.data.SubsetRandomSampler(train_sub_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, batch_size=512, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=1024, sampler=val_sampler)
        test_loader = DataLoader(dataset, batch_size=1024, sampler=test_sampler)

        # 初始化模型
        model = RelationAwareDDIPredictor(
            emb_dim=embeddings.shape[1], num_classes=num_classes
        ).to(device)

        # 计算类别权重
        train_labels = [dataset.labels[i] for i in train_sub_idx]
        label_counts = np.array([train_labels.count(i) for i in range(num_classes)])
        weights = 1.0 / label_counts
        weights = weights / weights.sum() * num_classes
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)

        criterion = ImprovedFocalLoss(alpha=class_weights, gamma=2.0, smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # 训练单个折
        best_model, fold_history = train_single_fold(
            model, train_loader, val_loader, embeddings, criterion, optimizer, epochs
        )

        # 测试该折
        fold_metrics_current, test_results = evaluate_single_fold(
            best_model, test_loader, embeddings, num_classes, fold,
            confusion_dir, ddi_source, curve_dir
        )

        fold_metrics.append(fold_metrics_current)
        all_fold_results.append(test_results)

        print(f"✅ 第 {fold} 折完成: "
              f"准确率 = {fold_metrics_current['accuracy']:.4f}, "
              f"Macro-F1 = {fold_metrics_current['macro_f1']:.4f}, "
              f"AUC-ROC = {fold_metrics_current['macro_auc_roc']:.4f}")

    # 绘制整体曲线
    print(f"📊 绘制整体ROC/PR曲线 - {ddi_source}")
    plot_all_curves(all_fold_results, num_classes, dataset.class_names, curve_dir, ddi_source)

    # 计算交叉验证统计结果
    cv_results = calculate_cv_statistics(fold_metrics)

    return cv_results, fold_metrics, all_fold_results


def train_single_fold(model, train_loader, val_loader, embeddings, criterion, optimizer, epochs):
    """训练单个折"""
    best_f1 = 0
    best_model_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_pairs, batch_labels, batch_features in train_loader:
            batch_pairs, batch_labels, batch_features = (
                batch_pairs.to(device), batch_labels.to(device), batch_features.to(device)
            )

            drug1 = embeddings[batch_pairs[:, 0]]
            drug2 = embeddings[batch_pairs[:, 1]]
            logits = model(drug1, drug2, batch_features)

            loss = criterion(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        # 验证
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for batch_pairs, batch_labels, batch_features in val_loader:
                batch_pairs, batch_labels, batch_features = (
                    batch_pairs.to(device), batch_labels.to(device), batch_features.to(device)
                )
                drug1 = embeddings[batch_pairs[:, 0]]
                drug2 = embeddings[batch_pairs[:, 1]]
                logits = model(drug1, drug2, batch_features)

                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(batch_labels.cpu().tolist())
                val_probs.extend(probs.cpu().tolist())

        val_f1 = f1_score(val_labels, val_preds, average='macro')

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()

        history.append({
            'epoch': epoch + 1,
            'train_loss': total_loss / len(train_loader),
            'val_f1': val_f1
        })

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    return model, history


def evaluate_single_fold(model, test_loader, embeddings, num_classes, fold, confusion_dir, ddi_source, curve_dir):
    """评估单个折"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch_pairs, batch_labels, batch_features in test_loader:
            batch_pairs, batch_labels, batch_features = (
                batch_pairs.to(device), batch_labels.to(device), batch_features.to(device)
            )
            drug1 = embeddings[batch_pairs[:, 0]]
            drug2 = embeddings[batch_pairs[:, 1]]
            logits = model(drug1, drug2, batch_features)

            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    # 计算所有指标
    metrics = comprehensive_evaluation(all_labels, all_preds, all_probs, num_classes)

    # 绘制混淆矩阵
    class_names = [f"Class {i}" for i in range(num_classes)]
    plot_confusion_matrix(all_labels, all_preds, class_names, fold, confusion_dir, ddi_source)

    # 绘制当前折的ROC和PR曲线
    plot_fold_curves(
        all_labels, all_preds, all_probs, num_classes, class_names, curve_dir,
        fold, ddi_source
    )

    return metrics, {
        'true_labels': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs
    }


def calculate_cv_statistics(fold_metrics):
    """计算交叉验证统计结果"""
    cv_results = {}

    for metric_name in fold_metrics[0].keys():
        # 处理嵌套指标
        if isinstance(fold_metrics[0][metric_name], (list, np.ndarray)):
            # 对每个类别分别计算统计量
            num_classes = len(fold_metrics[0][metric_name])
            for i in range(num_classes):
                class_metric_name = f"{metric_name}_class_{i}"
                class_values = [fold[metric_name][i] for fold in fold_metrics]

                cv_results[class_metric_name] = {
                    'mean': np.mean(class_values),
                    'std': np.std(class_values),
                    'values': class_values
                }
        else:
            # 标量指标
            metric_values = [fold[metric_name] for fold in fold_metrics]
            cv_results[metric_name] = {
                'mean': np.mean(metric_values),
                'std': np.std(metric_values),
                'values': metric_values
            }

    return cv_results


def print_cv_results(cv_results, fold_metrics, class_names, ddi_source):
    """打印交叉验证结果"""
    print(f"\n{'=' * 80}")
    print(f"🎯 {ddi_source} - 5折交叉验证最终结果")
    print(f"{'=' * 80}")

    # 主要指标表格
    main_metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_auc_roc', 'macro_auc_pr']
    print(f"\n📊 主要性能指标:")
    print(f"{'指标':<20} {'均值':<8} {'标准差':<8} {'各折结果'}")
    print(f"{'-' * 70}")

    for metric in main_metrics:
        mean_val = np.mean([fold[metric] for fold in fold_metrics])
        std_val = np.std([fold[metric] for fold in fold_metrics])
        fold_vals = [f'{fold[metric]:.4f}' for fold in fold_metrics]
        print(f"{metric:<20} {mean_val:.4f}   {std_val:.4f}   {fold_vals}")

    # 精确率召回率表格
    print(f"\n🎯 各类别精确率-召回率:")
    print(f"{'类别':<10} {'精确率':<8} {'召回率':<8}")
    print(f"{'-' * 30}")

    # for i, class_name in enumerate(class_names):
    #     precisions = [metrics['per_class_precision'][i] for metrics in fold_metrics]
    #     recalls = [metrics['per_class_recall'][i] for metrics in fold_metrics]
    #     print(f"{class_name:<10} {np.mean(precisions):.4f}   {np.mean(recalls):.4f}")
    num_classes = min(len(class_names), len(fold_metrics[0]['per_class_precision']))
    for i in range(num_classes):
        class_name = class_names[i]
        precisions = [metrics['per_class_precision'][i] for metrics in fold_metrics if
                      i < len(metrics['per_class_precision'])]
        recalls = [metrics['per_class_recall'][i] for metrics in fold_metrics if i < len(metrics['per_class_recall'])]
        print(f"{class_name:<10} {np.mean(precisions):.4f}   {np.mean(recalls):.4f}")


# =======================
# 主训练函数（多数据库支持）
# =======================
def train_improved_ddi_model(ddi_source="PDD Graph"):
    """支持多数据库(ogbl-ddi/DDInter/PDD Graph)的训练函数"""
    print(f"\n{'=' * 80}")
    print(f"🏁 开始处理数据库: {ddi_source}")
    print(f"{'=' * 80}")

    # 1. 创建结果目录
    result_dir = f"results/{ddi_source.replace(' ', '_')}"
    os.makedirs(result_dir, exist_ok=True)
    print(f"结果保存目录: {result_dir}")

    # 2. 加载预训练的药物嵌入
    print("=== 加载预训练药物嵌入 ===")
    data = torch.load("models/model/drug_global_embeddings_ablation_A5_3.pt")
    drug_ids = data["drug_ids"]
    embeddings = data["global_embeddings"].to(device)
    print(f"✅ 加载嵌入: {embeddings.shape}")

    # 3. 加载和处理DDI数据
    print(f"\n=== 加载 {ddi_source} DDI 数据 ===")

    # 定义数据库路径和规格
    DB_INFO = {
        "ogbl-ddi": {"path": "data/ogb_ddi/ogbl_ddi/mapping/test1.csv", "min_samples": 1000},
        "DDInter": {
            "path": "data/ddinter/test3.csv",
            "min_samples": 100,
            "label_map": {"Moderate": 0, "Minor": 1, "Major": 2}
        },
        "PDD Graph": {"path": "data/PDD_graph/test2.csv", "min_samples": 10}
    }

    if ddi_source not in DB_INFO:
        raise ValueError(f"❌ 未知数据库：{ddi_source}。支持：{list(DB_INFO.keys())}")

    config = DB_INFO[ddi_source]
    df = pd.read_csv(config["path"])

    # 打印原始数据信息
    print(f"原始DDI数据大小: {len(df)}")
    print(f"列名: {df.columns.tolist()}")

    # 4. 标准化的数据预处理流程 -------------------------------------------------
    def standardize_dataset(df, source):
        """统一不同数据库的格式"""
        # 列名标准化 (不同数据库可能有不同的列名)
        col_map = {
            "first drug id": "first drug id",  # PDD格式
            "second drug id": "second drug id",  # PDD格式
            "label": "label",  # PDD Graph格式
        }

        # 重命名列
        df.columns = [col_map.get(col.lower().replace(" ", "_").replace("-", "_"), col) for col in df.columns]

        # 确保有label列
        if "label" not in df.columns:
            if ddi_source in ["PDD Graph", "ogbl-ddi"]:
                # 对于PDD Graph，手动生成标签（二分类）
                print("🔄 为 {ddi_source} 生成二分类标签...")
                df["label"] = 1  # 所有现有样本为正例

                # 接下来添加负采样（保留您原来的处理逻辑）
                positives = df[["first drug id", "second drug id"]].copy()
                positives["label"] = 1
                drugs = list(pd.concat([df["first drug id"], df["second drug id"]]).unique())
                pos_set = set(zip(positives["first drug id"], positives["second drug id"]))

                # 采样负样本（比例1:1）
                num_negatives = len(positives)
                negatives = set()
                while len(negatives) < num_negatives:
                    d1, d2 = random.sample(drugs, 2)
                    if (d1, d2) not in pos_set and (d2, d1) not in pos_set:
                        negatives.add((d1, d2))

                negatives = pd.DataFrame(list(negatives), columns=["first drug id", "second drug id"])
                negatives["label"] = 0

                # 合并正负样本
                df = pd.concat([positives, negatives], ignore_index=True)
            else:
                raise ValueError("缺少标签列且不是PDD Graph数据库")

        return df

    # 应用标准化
    df = standardize_dataset(df, ddi_source)

    # 5. 标签处理逻辑 ---------------------------------------------------------
    # 特殊处理DDInter的字符串标签
    if ddi_source == "DDInter":
        # 检查是否有文本标签并转换为数字
        if df["label"].dtype == 'object':
            df["label"] = df["label"].map(config["label_map"])
            print("✅ DDInter文本标签映射完成")

    # 确保标签为数值类型
    if df["label"].dtype != int:
        try:
            df["label"] = pd.to_numeric(df["label"])
        except:
            # 如果没有明确标签映射，使用编码器
            print("⚠️ 检测到非数值标签，正在编码...")
            le = LabelEncoder()
            df["label"] = le.fit_transform(df["label"])
            print(f"标签编码映射：{dict(zip(le.classes_, range(len(le.classes_))))}")

    # 6. 过滤低频类别 -------------------------------------------------------
    label_counts = df["label"].value_counts()
    min_samples = config["min_samples"]  # 数据库特定的阈值

    # 对于二分类（PDD系列），只需过滤出现次数极少的情况
    if set(df["label"].unique()) == {0, 1}:
        valid_label_names = {0, 1}
        df_filtered = df
    else:
        # 多分类情况，过滤低频类
        valid_label_names = []
        for lbl, count in label_counts.items():
            if count >= min_samples or (ddi_source == "DDInter" and lbl in config["label_map"].values()):
                valid_label_names.append(lbl)
            else:
                print(f"过滤掉低频标签 {lbl} (仅出现 {count} 次)")

        df_filtered = df[df["label"].isin(valid_label_names)].copy()

    print(f"过滤后数据大小: {len(df_filtered)}")
    label_counts = df_filtered["label"].value_counts()
    print("过滤后类别分布:\n", label_counts)

    # 7. 创建统一标签映射 (保留原有的整数标签，避免重映射) ------------------------
    # labels_unique = sorted(df_filtered["label"].unique().tolist())  # 单独运行时候的解决方案，解决索引越界问题
    # valid_labels = sorted(valid_label_names)    # 避免排序造成映射错乱，保持与实际数据一致
    label_map = {lbl: i for i, lbl in enumerate(df_filtered["label"].unique())}  # 创建从原始标签到新索引的映射
    df_filtered["label"] = df_filtered["label"].map(label_map)  # 更新数据框中的标签
    if df_filtered["label"].isna().any():
        print(f"⚠️ 警告: {ddi_source} 存在未映射标签，已自动删除。")
        df_filtered = df_filtered.dropna(subset=["label"])

    num_classes = len(label_map)
    class_names = []

    # 特殊处理DDInter的类名
    if ddi_source == "DDInter":
        # 按标签值排序生成类名
        class_names = [
            "Moderate" if label_map.get(0) == i else
            "Minor" if label_map.get(1) == i else
            "Major" for i in range(num_classes)
        ]

    elif ddi_source == "ogbl-ddi":
        class_names = [f"Class {i}" for i in range(num_classes)]

    else:
        # 二分类命名
        if num_classes == 2:
            class_names = ["No Interaction", "Interaction"]
        else:
            class_names = [f"Class {i}" for i in range(num_classes)]

    print(f"类别数量: {num_classes}")
    print(f"类名列表: {class_names}")

    # 8. 创建数据集
    print("\n=== 创建训练数据集 ===")
    dataset = EnhancedDDIDataset(df_filtered, drug_ids, embeddings, label_map)
    dataset.class_names = class_names  # 添加类名属性以便后续使用

    if len(dataset) == 0:
        print("错误：没有有效的样本！")
        return

    # 9. 进行交叉验证
    cv_results, fold_metrics, all_fold_results = cross_validation_training(
        dataset, embeddings, num_classes, ddi_source, n_splits=5, epochs=50
    )

    # 10. 打印并保存结果
    print_cv_results(cv_results, fold_metrics, class_names, ddi_source)

    results = {
        'cv_results': cv_results,
        'fold_metrics': fold_metrics,
        'fold_results': all_fold_results,
        'class_names': class_names,
        'label_map': label_map,
        'source_db': ddi_source
    }

    save_path = f"{result_dir}/cross_validation_results.pth"
    torch.save(results, save_path)
    print(f"\n✅ {ddi_source} 交叉验证结果已保存到: {save_path}")

    # 11. 返回训练结果用于分析
    return results


# =======================
# 嵌入质量评估
# =======================
def evaluate_embedding_quality():
    """评估嵌入质量"""
    print("\n=== 嵌入质量评估 ===")
    data = torch.load("models/model/drug_global_embeddings_ablation_A5_3.pt")
    embeddings = data["global_embeddings"]

    print(f"嵌入形状: {embeddings.shape}")
    print(f"均值: {embeddings.mean().item():.4f}")
    print(f"标准差: {embeddings.std().item():.4f}")

    # 计算类内类间距离
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    normalized_emb = embeddings / norms
    similarity = torch.mm(normalized_emb, normalized_emb.t())

    # 排除对角线
    mask = torch.eye(len(embeddings)).bool()
    off_diag_similarity = similarity[~mask].view(len(embeddings), -1)

    print(f"平均相似度: {off_diag_similarity.mean().item():.4f}")
    print(f"相似度标准差: {off_diag_similarity.std().item():.4f}")


if __name__ == "__main__":
    # 评估嵌入质量
    evaluate_embedding_quality()

    # 数据库列表
    databases = [
        "ogbl-ddi",
        "PDD Graph",
        "DDInter"
    ]

    # 创建总结果目录
    if not os.path.exists("results"):
        os.makedirs("results")

    # 循环处理每个数据库
    results = {}
    for db in databases:
        try:
            db_results = train_improved_ddi_model(ddi_source=db)
            results[db] = db_results
        except Exception as e:
            print(f"❌ 处理数据库 {db} 时出错: {str(e)}")
            import traceback

            traceback.print_exc()

        print(f"\n{'=' * 80}")
        print(f"✅ 数据库 {db} 处理完成")
        print(f"{'=' * 80}\n\n")

    print("所有数据库处理完成！")
