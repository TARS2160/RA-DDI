import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from tqdm import tqdm
import pandas as pd
from dgl.nn.pytorch import SAGEConv
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------------
# 1. 加载数据
# ----------------------------
data = torch.load("models/model/drug_initial_embeddings.pt")
drug_ids = data["drug_ids"]
drug_features = data["embeddings"]
num_drugs, emb_dim = drug_features.shape
print(f"原始特征维度: {drug_features.shape}")

# 特征标准化
drug_features = F.normalize(drug_features, dim=1)

# ----------------------------
# 2. 构造图
# ----------------------------
kg_df = pd.read_csv("data/drugbank/drugbank_kg_triples_cleaned.csv")
drug2idx = {did: i for i, did in enumerate(drug_ids)}
src, dst = [], []
for _, row in kg_df.iterrows():
    h, t = row["head"], row["tail"]
    if h in drug2idx and t in drug2idx:
        src.append(drug2idx[h])
        dst.append(drug2idx[t])

g = dgl.graph((src + dst, dst + src), num_nodes=num_drugs)
print("Graph info:", g)


# ----------------------------
# 3. 简单的图传播（无训练）
# ----------------------------
def simple_graph_propagation(graph, features, num_propagation=2):
    """简单的图传播，不涉及对比学习"""
    print("进行图传播...")

    # 创建邻接矩阵（归一化）
    graph = dgl.add_self_loop(graph)  # 添加自环
    degs = graph.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    graph.ndata['norm'] = norm.unsqueeze(1)

    # 多轮传播
    current_features = features.clone()
    for i in range(num_propagation):
        print(f"传播轮次 {i + 1}/{num_propagation}")

        # 使用SAGEConv进行传播（不训练权重）
        conv = SAGEConv(emb_dim, emb_dim, 'mean').to(device)
        for param in conv.parameters():
            param.requires_grad = False  # 冻结参数

        graph_gpu = graph.to(device)
        features_gpu = current_features.to(device)

        with torch.no_grad():
            propagated = conv(graph_gpu, features_gpu)

        # 组合原始特征和传播特征
        alpha = 0.7  # 保留大部分原始信息
        current_features = alpha * features + (1 - alpha) * propagated.cpu()
        current_features = F.normalize(current_features, dim=1)

        del conv, graph_gpu, features_gpu
        if device == "cuda":
            torch.cuda.empty_cache()

    return current_features


# 执行图传播
print("开始图传播...")
global_embeddings = simple_graph_propagation(g, drug_features, num_propagation=3)

# 保存结果
torch.save({
    "drug_ids": drug_ids,
    "embeddings": global_embeddings
}, "models/model/drug_global_embeddings_simple.pt")

print("✅ 简单图传播embedding已保存，形状:", global_embeddings.shape)


# ----------------------------
# 4. 基于特征重构的图自编码器
# ----------------------------
class GraphAutoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.encoder = SAGEConv(in_dim, hidden_dim, 'mean')
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, graph, x):
        # 编码
        h = F.relu(self.encoder(graph, x))
        # 解码重构
        reconstructed = self.decoder(h)
        return reconstructed, h


def train_autoencoder(model, graph, features, epochs=20, batch_size=512):
    """训练图自编码器"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 选择训练节点
    train_indices = torch.randperm(num_drugs)[:min(3000, num_drugs)]

    for epoch in range(epochs):
        total_loss = 0
        indices = train_indices[torch.randperm(len(train_indices))]

        for i in range(0, len(indices), batch_size):
            batch_nodes = indices[i:i + batch_size]
            if len(batch_nodes) < 2:
                continue

            # 采样子图
            frontier = dgl.sampling.sample_neighbors(graph, batch_nodes, fanout=15)
            if frontier.num_edges() == 0:
                continue

            block = dgl.to_block(frontier, batch_nodes)
            src_features = features[block.srcdata[dgl.NID]]

            # 移动到GPU
            block = block.to(device)
            src_features = src_features.to(device)

            # 前向传播
            reconstructed, encoded = model(block, src_features)

            # 重构损失（只重构目标节点）
            target_features = features[batch_nodes].to(device)
            loss = F.mse_loss(reconstructed[:len(batch_nodes)], target_features)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(indices) // batch_size + 1)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Recon Loss: {avg_loss:.4f}")

    return model


# 训练自编码器
print("训练图自编码器...")
model_ae = GraphAutoencoder(emb_dim, 256, emb_dim).to(device)
model_ae = train_autoencoder(model_ae, g, drug_features)


# 生成编码后的特征
def encode_features(model, graph, features, batch_size=500):
    """编码所有特征"""
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, num_drugs, batch_size), desc="Encoding features"):
            batch_nodes = torch.arange(i, min(i + batch_size, num_drugs))

            frontier = dgl.sampling.sample_neighbors(graph, batch_nodes, fanout=10)
            if frontier.num_edges() == 0:
                frontier = dgl.graph((batch_nodes, batch_nodes), num_nodes=num_drugs)

            block = dgl.to_block(frontier, batch_nodes)
            batch_feat = features[block.srcdata[dgl.NID]].to(device)
            block = block.to(device)

            _, encoded = model(block, batch_feat)
            all_embeddings.append(encoded.cpu())

    return torch.cat(all_embeddings, dim=0)


print("生成自编码器embedding...")
ae_embeddings = encode_features(model_ae, g, drug_features)
ae_embeddings = F.normalize(ae_embeddings, dim=1)

torch.save({
    "drug_ids": drug_ids,
    "embeddings": ae_embeddings
}, "models/model/drug_global_embeddings_ae.pt")

print("✅ 自编码器embedding已保存，形状:", ae_embeddings.shape)


# ----------------------------
# 5. 评估特征质量
# ----------------------------
def evaluate_feature_quality(features, graph, k=5):
    """评估特征在图上的质量"""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    features_np = features.numpy()

    # 计算特征相似度
    feature_sim = cosine_similarity(features_np)

    # 计算图邻接关系
    src, dst = graph.edges()
    adj_matrix = np.zeros((num_drugs, num_drugs))
    adj_matrix[src.numpy(), dst.numpy()] = 1

    # 评估特征相似度与图结构的一致性
    hits = 0
    total = 0

    for i in range(num_drugs):
        # 基于特征的最近邻
        feat_neighbors = np.argsort(-feature_sim[i])[1:k + 1]  # 排除自身

        # 基于图的邻居
        graph_neighbors = np.where(adj_matrix[i] > 0)[0]

        if len(graph_neighbors) > 0:
            # 计算命中率
            common = len(set(feat_neighbors) & set(graph_neighbors))
            hits += common
            total += min(k, len(graph_neighbors))

    hit_rate = hits / total if total > 0 else 0
    print(f"特征-图结构命中率 (@{k}): {hit_rate:.4f}")
    return hit_rate


print("评估原始特征质量...")
original_hit_rate = evaluate_feature_quality(drug_features, g)

print("评估传播后特征质量...")
if 'global_embeddings' in locals():
    propagated_hit_rate = evaluate_feature_quality(global_embeddings, g)
    print(f"改进程度: {propagated_hit_rate - original_hit_rate:.4f}")