import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
from tqdm import tqdm
import csv
import os
import pickle

# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
NEIGHBOR_SAMPLE_SIZE = 4
EMBED_DIM = 256
NUM_LAYERS = 2


# ==================== 数据集 ====================
class DrugEmbeddingDataset(Dataset):
    def __init__(self, embedding_file, kg_file, entity2id):
        data = torch.load(embedding_file)
        self.drug_ids = data["drug_ids"]  # 有SMILES的药物DB ID
        self.embeddings = data["embeddings"]

        # 创建drug_id到索引的映射（基于entity2id的索引）
        self.drug_id_to_entity_idx = {}
        valid_drug_indices = []

        for i, drug_id in enumerate(self.drug_ids):
            if drug_id in entity2id:
                self.drug_id_to_entity_idx[drug_id] = entity2id[drug_id]
                valid_drug_indices.append(i)

        # 只保留在entity2id中存在的药物
        self.valid_drug_ids = [self.drug_ids[i] for i in valid_drug_indices]
        self.valid_embeddings = self.embeddings[valid_drug_indices]

        print(f"原始药物数量: {len(self.drug_ids)}, 有效药物数量: {len(self.valid_drug_ids)}")

        self.kg_triples = []
        with open(kg_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) < 3:
                    continue
                h, r = row[0], row[1]
                t = ",".join(row[2:])
                self.kg_triples.append((h, r, t))

    def __len__(self):
        return len(self.valid_drug_ids)

    def __getitem__(self, idx):
        return self.valid_drug_ids[idx], self.valid_embeddings[idx]


# # ==================== 邻居采样 ====================
# class NeighborSampler:
#     def __init__(self, neighbor_dict, drug_id_to_entity_idx):
#         self.neighbor_dict = neighbor_dict
#         self.drug_id_to_entity_idx = drug_id_to_entity_idx
#
#     def sample(self, drug_ids, sample_size):
#         adj_entity = []
#         adj_relation = []
#         edge_weights = []
#
#         for drug_id in drug_ids:
#             # 使用entity2id中的索引
#             entity_idx = self.drug_id_to_entity_idx.get(drug_id)
#             if entity_idx is None:
#                 # 如果药物不在entity2id中，使用全0邻居
#                 adj_entity.append([0] * sample_size)
#                 adj_relation.append([0] * sample_size)
#                 edge_weights.append([0.0] * sample_size)
#                 continue
#
#             neighbors = self.neighbor_dict.get(entity_idx, [])
#
#             # 确保邻居列表长度一致
#             if len(neighbors) == 0:
#                 neighbors = [(0, entity_idx)]  # 自环
#
#             # 采样或填充到指定大小
#             if len(neighbors) >= sample_size:
#                 sampled_neighbors = random.sample(neighbors, sample_size)
#             else:
#                 # 如果邻居不够，重复采样
#                 sampled_neighbors = neighbors
#                 while len(sampled_neighbors) < sample_size:
#                     remaining = sample_size - len(sampled_neighbors)
#                     sampled_neighbors.extend(random.choices(neighbors, k=min(remaining, len(neighbors))))
#
#             adj_entity.append([n[1] for n in sampled_neighbors])
#             adj_relation.append([n[0] for n in sampled_neighbors])
#             edge_weights.append([1.0] * len(sampled_neighbors))
#
#         # 验证所有列表长度一致
#         assert all(len(lst) == sample_size for lst in adj_entity), "adj_entity长度不一致"
#         assert all(len(lst) == sample_size for lst in adj_relation), "adj_relation长度不一致"
#         assert all(len(lst) == sample_size for lst in edge_weights), "edge_weights长度不一致"
#
#         return (torch.tensor(adj_entity, dtype=torch.long, device=DEVICE),
#                 torch.tensor(adj_relation, dtype=torch.long, device=DEVICE),
#                 torch.tensor(edge_weights, dtype=torch.float, device=DEVICE))
#
#
# def build_neighbor_dict(kg_triples, entity2id, relation2id):
#     neighbor_dict = defaultdict(list)
#     for h, r, t in kg_triples:
#         if h in entity2id and t in entity2id and r in relation2id:
#             h_idx = entity2id[h]
#             t_idx = entity2id[t]
#             r_idx = relation2id[r]
#             neighbor_dict[h_idx].append((r_idx, t_idx))
#             neighbor_dict[t_idx].append((r_idx, h_idx))  # 双向
#
#     return neighbor_dict

#消融实验 A8，验证多条邻居
# ==================== 邻居采样 ====================
# class NeighborSampler:
#     def __init__(self, neighbor_dict, drug_id_to_entity_idx):
#         self.neighbor_dict = neighbor_dict
#         self.drug_id_to_entity_idx = drug_id_to_entity_idx
#
#     def sample(self, drug_ids, sample_size, hops=1):
#         adj_entity = []
#         adj_relation = []
#         edge_weights = []
#
#         for drug_id in drug_ids:
#             entity_idx = self.drug_id_to_entity_idx.get(drug_id)
#             if entity_idx is None:
#                 adj_entity.append([0] * sample_size)
#                 adj_relation.append([0] * sample_size)
#                 edge_weights.append([0.0] * sample_size)
#                 continue
#
#             # ---------- ✅ 多跳采样逻辑 ----------
#             neighbors = set()
#             frontier = [entity_idx]
#             for h in range(hops):
#                 next_frontier = []
#                 for ent in frontier:
#                     neighs = self.neighbor_dict.get(ent, [])
#                     for r, t in neighs:
#                         neighbors.add((r, t))
#                         next_frontier.append(t)
#                 frontier = next_frontier
#
#             # 如果一个都没找到，添加自环
#             if len(neighbors) == 0:
#                 neighbors = [(0, entity_idx)]
#
#             neighbors = list(neighbors)
#
#             # ---------- 采样或填充 ----------
#             if len(neighbors) >= sample_size:
#                 sampled_neighbors = random.sample(neighbors, sample_size)
#             else:
#                 sampled_neighbors = neighbors
#                 while len(sampled_neighbors) < sample_size:
#                     remaining = sample_size - len(sampled_neighbors)
#                     sampled_neighbors.extend(
#                         random.choices(neighbors, k=min(remaining, len(neighbors)))
#                     )
#
#             adj_entity.append([n[1] for n in sampled_neighbors])
#             adj_relation.append([n[0] for n in sampled_neighbors])
#             edge_weights.append([1.0] * sample_size)
#
#         assert all(len(lst) == sample_size for lst in adj_entity), "adj_entity长度不一致"
#         assert all(len(lst) == sample_size for lst in adj_relation), "adj_relation长度不一致"
#         assert all(len(lst) == sample_size for lst in edge_weights), "edge_weights长度不一致"
#
#         return (
#             torch.tensor(adj_entity, dtype=torch.long, device=DEVICE),
#             torch.tensor(adj_relation, dtype=torch.long, device=DEVICE),
#             torch.tensor(edge_weights, dtype=torch.float, device=DEVICE)
#         )
    #消融实验 A8_1，解决3跳采样MemoryError
    # def sample(self, drug_ids, sample_size, hops=1, per_hop_limit=32):
    #     adj_entity = []
    #     adj_relation = []
    #     edge_weights = []
    #
    #     for drug_id in drug_ids:
    #         entity_idx = self.drug_id_to_entity_idx.get(drug_id)
    #         if entity_idx is None:
    #             adj_entity.append([0] * sample_size)
    #             adj_relation.append([0] * sample_size)
    #             edge_weights.append([0.0] * sample_size)
    #             continue
    #
    #         # multi-hop neighbor collection
    #         neighbors = set()
    #         frontier = [entity_idx]
    #         for h in range(hops):
    #             next_frontier = []
    #             for ent in frontier:
    #                 neighs = self.neighbor_dict.get(ent, [])
    #                 if len(neighs) > per_hop_limit:
    #                     neighs = random.sample(neighs, per_hop_limit)
    #                 for r, t in neighs:
    #                     neighbors.add((r, t))
    #                     next_frontier.append(t)
    #             frontier = next_frontier
    #
    #         neighbors = list(neighbors)
    #         if len(neighbors) == 0:
    #             neighbors = [(0, entity_idx)]
    #
    #         # sample from total neighbors
    #         if len(neighbors) >= sample_size:
    #             sampled_neighbors = random.sample(neighbors, sample_size)
    #         else:
    #             sampled_neighbors = neighbors * (sample_size // len(neighbors) + 1)
    #             sampled_neighbors = sampled_neighbors[:sample_size]
    #
    #         adj_entity.append([n[1] for n in sampled_neighbors])
    #         adj_relation.append([n[0] for n in sampled_neighbors])
    #         edge_weights.append([1.0] * len(sampled_neighbors))
    #
    #     return (torch.tensor(adj_entity, dtype=torch.long, device=DEVICE),
    #             torch.tensor(adj_relation, dtype=torch.long, device=DEVICE),
    #             torch.tensor(edge_weights, dtype=torch.float, device=DEVICE))

#消融实验 A8_2,解决越界问题
# ==================== 邻居采样（支持多跳 & 边界防护） ====================
class NeighborSampler:
    def __init__(self, neighbor_dict, drug_id_to_entity_idx, num_entities):
        """
        neighbor_dict: {entity_idx: [(rel_idx, target_entity_idx), ...], ...}
        drug_id_to_entity_idx: {drug_id: entity_idx, ...}
        num_entities: 总实体数（用于 clamp 边界检查）
        """
        self.neighbor_dict = neighbor_dict
        self.drug_id_to_entity_idx = drug_id_to_entity_idx
        self.num_entities = int(num_entities)

    def sample(self, drug_ids, sample_size, hops=1, per_hop_limit=32):
        """
        返回: (adj_entity_tensor, adj_relation_tensor, edge_weights_tensor)
        - adj_entity_tensor: shape [len(drug_ids), sample_size]
        - 所有索引保证在 [0, num_entities-1] 范围内（越界映射到 0）
        """
        adj_entity = []
        adj_relation = []
        edge_weights = []

        for drug_id in drug_ids:
            entity_idx = self.drug_id_to_entity_idx.get(drug_id)
            if entity_idx is None:
                adj_entity.append([0] * sample_size)
                adj_relation.append([0] * sample_size)
                edge_weights.append([0.0] * sample_size)
                continue

            # 多跳采样（每跳限制 per_hop_limit，防止爆炸）
            neighbors = set()
            frontier = [entity_idx]

            for _ in range(hops):
                next_frontier = []
                for ent in frontier:
                    neighs = self.neighbor_dict.get(ent, [])
                    if len(neighs) > per_hop_limit:
                        neighs = random.sample(neighs, per_hop_limit)
                    for r, t in neighs:
                        # 只收集目标 t（整数）及关系 r
                        # 但是不在此处 assume t 一定小于 num_entities（会在后面保护）
                        neighbors.add((int(r), int(t)))
                        next_frontier.append(int(t))
                frontier = next_frontier

                # 防止 neighbors 过大导致内存问题
                if len(neighbors) > sample_size * 50:
                    break

            neighbors = list(neighbors)
            if len(neighbors) == 0:
                neighbors = [(0, entity_idx)]

            # 采样或填充到固定大小
            if len(neighbors) >= sample_size:
                sampled_neighbors = random.sample(neighbors, sample_size)
            else:
                # 重复元素以填满
                times = sample_size // len(neighbors) + 1
                sampled_neighbors = (neighbors * times)[:sample_size]

            # 提取并做边界检查（将越界的 idx 映射为 0）
            entities = []
            relations = []
            for r, t in sampled_neighbors:
                if t < 0 or t >= self.num_entities:
                    # 映射到 0（或你想要的安全占位 idx）
                    t_safe = 0
                else:
                    t_safe = t
                entities.append(int(t_safe))
                relations.append(int(r) if isinstance(r, int) else 0)

            adj_entity.append(entities)
            adj_relation.append(relations)
            edge_weights.append([1.0] * sample_size)

        # 转为 tensor 并返回
        return (
            torch.tensor(adj_entity, dtype=torch.long, device=DEVICE),
            torch.tensor(adj_relation, dtype=torch.long, device=DEVICE),
            torch.tensor(edge_weights, dtype=torch.float, device=DEVICE)
        )


def build_neighbor_dict(kg_triples, entity2id, relation2id):
    neighbor_dict = defaultdict(list)
    for h, r, t in kg_triples:
        if h in entity2id and t in entity2id and r in relation2id:
            h_idx = entity2id[h]
            t_idx = entity2id[t]
            r_idx = relation2id[r]
            neighbor_dict[h_idx].append((r_idx, t_idx))
            neighbor_dict[t_idx].append((r_idx, h_idx))  # 双向

    return neighbor_dict

# ==================== 模型 ====================
# class RelationalAttentionRGCN(nn.Module):
#     def __init__(self, num_entities, num_relations, embed_dim, num_layers):
#         super(RelationalAttentionRGCN, self).__init__()
#         self.num_layers = num_layers
#         self.embed_dim = embed_dim
#
#         # 实体嵌入层，用于处理KG中的所有实体
#         self.entity_embedding = nn.Embedding(num_entities, embed_dim)
#         nn.init.xavier_uniform_(self.entity_embedding.weight)
#
#         self.rgcn_layers = nn.ModuleList([
#             RGCNAggregator(embed_dim, num_relations) for _ in range(num_layers)
#         ])
#         # 消融实验 A6
#         # self.rgcn_layers = nn.ModuleList([
#         #     SimpleGCNAggregator(embed_dim) for _ in range(num_layers)
#         # ])
#         self.attention = nn.MultiheadAttention(embed_dim, num_heads=4)
#
#     def forward(self, drug_emb, drug_entity_indices, adj_entity, adj_relation, edge_weights):
#         # 将初始嵌入与KG实体嵌入结合
#         entity_emb = self.entity_embedding(drug_entity_indices)
#         combined_emb = drug_emb + entity_emb  # 残差连接
#
#         x = combined_emb.unsqueeze(0)
#         for layer in self.rgcn_layers:
#             x = layer(x, adj_entity, adj_relation, edge_weights)
#         attn_output, _ = self.attention(x, x, x)
#         return attn_output.squeeze(0)

# 按层采样，保证不同层访问不同hop领域，捕获多阶语义关系1，修改之后除了1-hop和2-hop的区别之外两层的逻辑结构没有任何区别
# class RelationalAttentionRGCN(nn.Module):
#     def __init__(self, num_entities, num_relations, embed_dim, num_layers):
#         super(RelationalAttentionRGCN, self).__init__()
#         self.num_layers = num_layers
#         self.embed_dim = embed_dim
#
#         # 实体嵌入层，用于处理KG中的所有实体
#         self.entity_embedding = nn.Embedding(num_entities, embed_dim)
#         nn.init.xavier_uniform_(self.entity_embedding.weight)
#
#         self.rgcn_layers = nn.ModuleList([
#             RGCNAggregator(embed_dim, num_relations) for _ in range(num_layers)
#         ])
#
#         # 多头注意力：我们将对每层的输出做 attention 再融合
#         self.attention = nn.MultiheadAttention(embed_dim, num_heads=4)
#
#     def forward(self,
#                 drug_emb=None,
#                 drug_entity_indices=None,
#                 adj_entity=None,
#                 adj_relation=None,
#                 edge_weights=None,
#                 neighbor_sampler=None):
#         """
#         支持两种调用方式：
#         1) 外部提供 drug_emb (float tensor of shape [B, D]) 和外部采样的 adj_entity/... -> model 做聚合
#            调用方式（你原来的）: model(emb, drug_entity_indices, adj_entity, adj_relation, edge_weights)
#         2) 传入 neighbor_sampler 由模型按层动态采样（multi-hop per-layer）:
#            调用方式: model(drug_entity_indices, neighbor_sampler=neighbor_sampler)
#         返回: tensor [B, D] （每个药物的全局 embedding）
#         """
#         # --- 准备 combined_emb ---
#         # 如果 drug_emb 提供，直接使用它（通常是float tensor）
#         if drug_emb is not None:
#             combined_emb = drug_emb  # [B, D], float
#         else:
#             # 否则从 entity_embedding 查表（需要 drug_entity_indices）
#             if drug_entity_indices is None:
#                 raise ValueError("Either drug_emb or drug_entity_indices must be provided.")
#             # ensure long dtype for embedding lookup
#             if drug_entity_indices.dtype != torch.long:
#                 drug_entity_indices = drug_entity_indices.long()
#             device = next(self.entity_embedding.parameters()).device
#             drug_entity_indices = drug_entity_indices.to(device)
#             combined_emb = self.entity_embedding(drug_entity_indices)  # [B, D]
#
#         # 保证 tensor 在正确设备
#         combined_emb = combined_emb.to(next(self.entity_embedding.parameters()).device)
#
#         # 变换为 attention 要求的 (seq_len, batch, dim) 格式，这里先把 node emb 做成 [1, B, D]
#         x = combined_emb.unsqueeze(0)  # [1, B, D]
#
#         # --- 遍历每层 RGCN，支持按层动态采样或使用外部邻接 ---
#         layer_outputs = []
#         for l, layer in enumerate(self.rgcn_layers):
#             if neighbor_sampler is not None:
#                 # neighbor_sampler.sample expects drug ids as python iter/list or tensor of ids used in its mapping
#                 # 如果用户传入的是 drug_entity_indices，我们优先用它；否则尝试从 combined_emb 的批次长度推断
#                 sampler_input = drug_entity_indices if drug_entity_indices is not None else None
#                 # 注意：neighbor_sampler.sample 的第一个参数在你的实现里是 drug_ids（原始 id 列表/张量）
#                 # 我们传入 sampler_input（通常是 dataset 中转换后的 entity idx 列表）
#                 adj_entity_l, adj_relation_l, edge_weights_l = neighbor_sampler.sample(
#                     sampler_input,
#                     sample_size=64,
#                     hops=l + 1,
#                     per_hop_limit=16
#                 )
#             else:
#                 # 使用外部传入的邻接（相同邻接用于每层）
#                 if adj_entity is None or adj_relation is None or edge_weights is None:
#                     raise ValueError("adj_entity/adj_relation/edge_weights must be provided if neighbor_sampler is None.")
#                 # 保证在正确的 device
#                 adj_entity_l = adj_entity.to(next(self.entity_embedding.parameters()).device)
#                 adj_relation_l = adj_relation.to(next(self.entity_embedding.parameters()).device)
#                 edge_weights_l = edge_weights.to(next(self.entity_embedding.parameters()).device)
#
#             # 调用 RGCN 层聚合；layer 期望 node_emb 形状 [1, B, D]
#             x = layer(x, adj_entity_l, adj_relation_l, edge_weights_l)  # 返回 [1, B, D]
#             layer_outputs.append(x)
#
#         # --- 使用 MultiheadAttention 融合各层输出 ---
#         # 将多个层的输出堆叠为 seq_len = num_layers
#         # layer_outputs: list of [1, B, D] -> stack -> [num_layers, B, D]
#         attn_input = torch.cat(layer_outputs, dim=0)  # [num_layers, B, D]
#
#         # MultiheadAttention expects (seq_len, batch, embed)
#         attn_output, _ = self.attention(attn_input, attn_input, attn_input)
#         # attn_output: [num_layers, B, D]
#         # 对层维度做平均融合为最终 embedding
#         fused = attn_output.mean(dim=0)  # [B, D]
#
#         return fused  # [B, D]
class RelationalAttentionRGCN(nn.Module):
    """
    Compatible with user-provided NeighborSampler.sample(...) -> (adj_entity[B,S], adj_relation[B,S], edge_weights[B,S])
    and RGCNAggregator (expects node_emb [1,B,D], adj_entity [B,S], adj_relation [B,S], edge_weights [B,S]).

    - Layer0: uniform 1-hop sample -> normalization + RGCNAggregator
    - Layer1: candidate 2-hop sample -> neighbor-level scoring via small MLP (center, neighbor, relation) -> top-K selection or weighted softmax -> RGCNAggregator
    - Layer outputs normalized, fused by MultiHeadAttention and returned as [B, D]
    """
    def __init__(self,
                 num_entities,
                 num_relations,
                 embed_dim=256,
                 num_layers=2,
                 layer0_sample_size=64,
                 layer1_sample_size=64,
                 layer1_candidate_size=256,
                 per_hop_limit=16,
                 dropout=0.1,
                 topk_selection=True):
        super().__init__()

        assert num_layers >= 2, "This implementation expects at least 2 layers (we use layer0 and layer1)."
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_relations = num_relations
        self.num_entities = num_entities
        self.device = DEVICE

        # shared entity embedding (will be set into RGCNAggregator via its .set_entity_embedding)
        self.entity_embedding = nn.Embedding(num_entities, embed_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight)

        # relation embedding for computing neighbor-level attention
        self.relation_embedding = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

        # RGCN layers: use existing RGCNAggregator instances (constructed outside or here)
        self.rgcn_layers = nn.ModuleList([
            RGCNAggregator(embed_dim, num_relations) for _ in range(num_layers)
        ])
        # inject entity_embedding into RGCNAggregator instances (they expose set_entity_embedding)
        for layer in self.rgcn_layers:
            try:
                layer.set_entity_embedding(self.entity_embedding)
            except Exception as e:
                # if user's RGCNAggregator has different interface, raise informative error
                raise RuntimeError("RGCNAggregator must implement set_entity_embedding(entity_embedding)") from e

        # small MLP to score neighbor given (center, neighbor, relation)
        # input dim = 3 * D + (optionally) elementwise product -> we'll use concat of (h_c, h_n, r_e, h_c * h_n)
        score_in_dim = embed_dim * 4
        self.score_mlp = nn.Sequential(
            nn.Linear(score_in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        # per-layer linear projections + norm + dropout to stabilize and avoid collapse
        self.layer_projs = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.layer_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        # fusion attention across layers (seq_len=num_layers)
        # use batch_first=False since MultiheadAttention expects (L, B, D) normally; but our attention below will handle either.
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=False)

        # sampling/selection hyperparams
        self.layer0_sample_size = int(layer0_sample_size)
        self.layer1_sample_size = int(layer1_sample_size)
        self.layer1_candidate_size = int(layer1_candidate_size)
        self.per_hop_limit = int(per_hop_limit)
        self.topk_selection = bool(topk_selection)  # if False, will use soft weights (no hard topk)

        # final layernorm
        self.final_norm = nn.LayerNorm(embed_dim)

        self.to(self.device)

    def _ensure_device(self, t):
        return t.to(self.device)

    def _compute_row_normalized_weights(self, raw_weights):
        # raw_weights: [B, S] (float)
        # produce normalized weights with safe division (avoid zero rows)
        eps = 1e-8
        row_sum = raw_weights.sum(dim=1, keepdim=True)  # [B,1]
        row_sum = row_sum + eps
        return raw_weights / row_sum

    def _score_neighbors(self, center_emb, neighbor_emb, rel_emb):
        """
        center_emb: [B, 1, D] or [B, D] -> we'll expand appropriately
        neighbor_emb: [B, C, D]
        rel_emb: [B, C, D]
        returns: scores [B, C]
        """
        # Ensure shapes
        if center_emb.dim() == 2:
            center = center_emb.unsqueeze(1)  # [B,1,D]
        else:
            center = center_emb  # [B,1,D]
        B, C, D = neighbor_emb.shape

        center_rep = center.expand(-1, C, -1)  # [B, C, D]
        # elementwise product
        prod = center_rep * neighbor_emb
        # concat [center, neighbor, rel, prod] -> [B, C, 4D]
        cat = torch.cat([center_rep, neighbor_emb, rel_emb, prod], dim=-1)  # [B,C,4D]
        flat = cat.view(B * C, 4 * D)
        scores = self.score_mlp(flat)  # [B*C,1]
        scores = scores.view(B, C)  # [B, C]
        return scores

    def forward(self,
                drug_emb=None,
                drug_entity_indices=None,
                adj_entity=None,
                adj_relation=None,
                edge_weights=None,
                neighbor_sampler=None):
        """
        Two supported modes (like original):
          - Provide drug_emb (float tensor [B,D]) -> use those as initial node states
          - Provide drug_entity_indices (LongTensor [B]) -> lookup entity_embedding
        neighbor_sampler: must be your NeighborSampler instance (unchanged)
        """

        # prepare combined_emb: [B, D]
        if drug_emb is not None:
            combined_emb = drug_emb.to(self.device)
        else:
            if drug_entity_indices is None:
                raise ValueError("Either drug_emb or drug_entity_indices must be provided.")
            if drug_entity_indices.dtype != torch.long:
                drug_entity_indices = drug_entity_indices.long()
            combined_emb = self.entity_embedding(drug_entity_indices.to(self.device))  # [B, D]

        combined_emb = combined_emb.to(self.device)
        B = combined_emb.size(0)

        # shape for RGCN input: node_emb [1, B, D]
        x = combined_emb.unsqueeze(0)  # [1, B, D]

        layer_outputs = []

        # iterate layers
        for l in range(self.num_layers):
            # get adjacency for this layer
            if neighbor_sampler is None:
                # fallback to provided static adj (not recommended)
                if adj_entity is None or adj_relation is None or edge_weights is None:
                    raise ValueError("When neighbor_sampler is None, adj_entity/adj_relation/edge_weights must be provided.")
                adj_entity_l = adj_entity.to(self.device)
                adj_relation_l = adj_relation.to(self.device)
                edge_weights_l = edge_weights.to(self.device)
            else:
                # sampler expects python list of drug ids / entity idx; pass entity indices
                sampler_input = drug_entity_indices.cpu().tolist() if isinstance(drug_entity_indices, torch.Tensor) else drug_entity_indices

                if l == 0:
                    # Layer 0: uniform 1-hop sample (as your existing sampler does)
                    adj_entity_l, adj_relation_l, edge_weights_l = neighbor_sampler.sample(
                        sampler_input,
                        sample_size=self.layer0_sample_size,
                        hops=1,
                        per_hop_limit=self.per_hop_limit
                    )
                    # sampler returns CPU tensors or device-specific; move to model device
                    adj_entity_l = adj_entity_l.to(self.device)
                    adj_relation_l = adj_relation_l.to(self.device)
                    edge_weights_l = edge_weights_l.to(self.device)

                    # normalize weights row-wise (edge_weights_l likely all ones)
                    edge_weights_l = self._compute_row_normalized_weights(edge_weights_l)

                    # optional: small attention reweighting using degree or relation frequency could be added here
                else:
                    # Layer 1: candidate 2-hop sample then compute neighbor-level scores using learned relation embedding & neighbor embedding
                    adj_entity_cand, adj_relation_cand, _ = neighbor_sampler.sample(
                        sampler_input,
                        sample_size=self.layer1_candidate_size,
                        hops=2,
                        per_hop_limit=self.per_hop_limit
                    )
                    adj_entity_cand = adj_entity_cand.to(self.device)
                    adj_relation_cand = adj_relation_cand.to(self.device)

                    # lookup neighbor embeddings and relation embeddings
                    # neighbor_emb_cand: [B, C, D]
                    neighbor_emb_cand = self.entity_embedding(adj_entity_cand)  # [B, C, D]
                    rel_emb_cand = self.relation_embedding(adj_relation_cand)  # [B, C, D]

                    # center emb: [B, D] -> unsqueeze as [B,1,D]
                    center_emb = combined_emb  # [B, D]

                    # compute raw scores [B, C]
                    raw_scores = self._score_neighbors(center_emb, neighbor_emb_cand, rel_emb_cand)

                    if self.topk_selection:
                        # pick top-K by raw_scores per row
                        K = self.layer1_sample_size
                        C = raw_scores.size(1)
                        if C < K:
                            # repeat to fill if less candidates than K
                            times = (K // C) + 1
                            adj_entity_cand = adj_entity_cand.repeat(1, times)[:, :K]
                            adj_relation_cand = adj_relation_cand.repeat(1, times)[:, :K]
                            neighbor_emb_cand = neighbor_emb_cand.repeat(1, times, 1)[:, :K, :]
                            rel_emb_cand = rel_emb_cand.repeat(1, times, 1)[:, :K, :]
                            raw_scores = raw_scores.repeat(1, times)[:, :K]
                            C = K

                        topk_scores, topk_idx = torch.topk(raw_scores, k=K, dim=1)
                        batch_idx = torch.arange(0, B, device=self.device).unsqueeze(1)
                        adj_entity_l = adj_entity_cand[batch_idx, topk_idx]  # [B, K]
                        adj_relation_l = adj_relation_cand[batch_idx, topk_idx]  # [B, K]
                        # compute normalized soft weights across topk
                        edge_weights_l = F.softmax(topk_scores, dim=1)  # [B, K]
                    else:
                        # use softmax across all candidates (no hard topk)
                        edge_weights_all = F.softmax(raw_scores, dim=1)  # [B, C]
                        # if C > desired S, we can sample proportional to weights or keep all and let aggregator handle (we choose to pick top-K by weight)
                        K = self.layer1_sample_size
                        C = edge_weights_all.size(1)
                        topk_scores, topk_idx = torch.topk(edge_weights_all, k=min(K, C), dim=1)
                        batch_idx = torch.arange(0, B, device=self.device).unsqueeze(1)
                        adj_entity_l = adj_entity_cand[batch_idx, topk_idx]
                        adj_relation_l = adj_relation_cand[batch_idx, topk_idx]
                        edge_weights_l = topk_scores  # already normalized
                    # now adj_entity_l: [B, S2], adj_relation_l: [B, S2], edge_weights_l: [B, S2]

            # --- call RGCN aggregator for this layer ---
            # aggregator expects node_emb shape [1, B, D], adj_entity [B,S], adj_relation [B,S], edge_weights [B,S]
            layer_module = self.rgcn_layers[l]
            # call aggregator (it returns [1,B,D])
            x = layer_module(x, adj_entity_l, adj_relation_l, edge_weights_l)

            # post-process layer output: x [1,B,D] -> squeeze -> [B,D]
            x_squeezed = x.squeeze(0)

            # projection, dropout, norm, residual with previous combined embedding to stabilize
            proj = self.layer_projs[l](x_squeezed)  # [B,D]
            proj = self.layer_dropouts[l](proj)
            proj = self.layer_norms[l](proj + combined_emb)  # residual to prevent collapse

            # update combined_emb for next layer propagation (so second layer sees transformed center)
            combined_emb = proj  # [B,D]

            # push into shape expected for next aggregator: [1,B,D]
            x = combined_emb.unsqueeze(0)

            layer_outputs.append(x)  # keep [1,B,D] elements

        # fuse via multihead attention over layers
        attn_input = torch.cat(layer_outputs, dim=0)  # [L, B, D]
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        fused = attn_output.mean(dim=0)  # [B, D]
        fused = self.final_norm(fused)

        return fused  # [B, D]


class RGCNAggregator(nn.Module):
# 消融实验 A6
# class SimpleGCNAggregator(nn.Module):
    def __init__(self, embed_dim, num_relations):
        super(RGCNAggregator, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_relations, embed_dim, embed_dim))
        self.residual = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.W)

        # 在RGCNAggregator中也添加entity_embedding
        self.entity_embedding = None

    # 消融实验 A6,把原实验改为共享矩阵
    # def __init__(self, embed_dim):
    #     super(SimpleGCNAggregator, self).__init__()
    #     self.W_shared = nn.Linear(embed_dim, embed_dim)
    #     self.residual = nn.Linear(embed_dim, embed_dim)
    #     self.entity_embedding = None

    def set_entity_embedding(self, entity_embedding):
        self.entity_embedding = entity_embedding

    def forward(self, node_emb, adj_entity, adj_relation, edge_weights):
        batch_size = node_emb.size(1)
        neighbor_msgs = torch.zeros_like(node_emb)

        for rel in range(self.W.size(0)):
            mask = (adj_relation == rel).unsqueeze(-1).float()
            # 使用entity_embedding来获取邻居嵌入
            neighbor_embed = self.entity_embedding(adj_entity)
            weighted_neighbors = neighbor_embed @ self.W[rel]
            # 使用sum而不是mean，因为mask已经处理了无效值
            neighbor_msgs += (weighted_neighbors * mask * edge_weights.unsqueeze(-1)).sum(dim=1).unsqueeze(0)

        return F.relu(node_emb + neighbor_msgs + self.residual(node_emb))
    # 消融实验 A6, 替换按关系变换为对邻居嵌入使用共享线性
    # def forward(self, node_emb, adj_entity, adj_relation, edge_weights):
    #     # 忽略关系类型，仅做普通邻居聚合
    #     neighbor_embed = self.entity_embedding(adj_entity)  # [B, N, D]
    #     # 聚合邻居（加权平均）
    #     neighbor_mean = (neighbor_embed * edge_weights.unsqueeze(-1)).mean(dim=1)
    #     out = F.relu(self.W_shared(neighbor_mean) + self.residual(node_emb.squeeze(0)))
    #     return out.unsqueeze(0)

# ==================== 工具函数 ====================
def load_mapping(file_path, limit=None):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"加载 {os.path.basename(file_path)}")):
            if limit and i >= limit:
                break
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            k, v = parts
            mapping[k] = int(v)
    return mapping


# ==================== 主训练 ====================
def train_global_embeddings(embeddings_file, kg_file, output_file, entity2id_file, relation2id_file):
    # 加载所有实体映射，不限制数量
    entity2id = load_mapping(entity2id_file, limit=None)
    relation2id = load_mapping(relation2id_file)

    num_entities = len(entity2id)
    num_relations = len(relation2id)
    print(f"实体数量: {num_entities}, 关系数量: {num_relations}")

    dataset = DrugEmbeddingDataset(embeddings_file, kg_file, entity2id)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    neighbor_dict = build_neighbor_dict(dataset.kg_triples, entity2id, relation2id)
    # neighbor_sampler = NeighborSampler(neighbor_dict, dataset.drug_id_to_entity_idx)
    # 消融实验 A8_2
    neighbor_sampler = NeighborSampler(neighbor_dict, dataset.drug_id_to_entity_idx, num_entities=num_entities)
    model = RelationalAttentionRGCN(
        num_entities=num_entities,
        num_relations=num_relations,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    # 设置entity_embedding到RGCNAggregator层
    for layer in model.rgcn_layers:
        layer.set_entity_embedding(model.entity_embedding)

    model.eval()

    global_embeddings = []
    with torch.no_grad():
        for drug_ids, emb in tqdm(dataloader, desc="Generating global embeddings"):
            emb = emb.to(DEVICE)

            # 获取药物在entity2id中的索引
            # drug_entity_indices = torch.tensor([
            #     dataset.drug_id_to_entity_idx[drug_id] for drug_id in drug_ids
            # ], device=DEVICE)

            # 按层采样，保证不同层访问不同hop领域，捕获多阶语义关系2
            drug_entity_indices = torch.tensor(
                [dataset.drug_id_to_entity_idx[drug_id] for drug_id in drug_ids],
                dtype=torch.long,
                device=DEVICE
            )

            # adj_entity, adj_relation, edge_weights = neighbor_sampler.sample(drug_ids, NEIGHBOR_SAMPLE_SIZE)

            # 消融实验 A8
            # adj_entity, adj_relation, edge_weights = neighbor_sampler.sample(
            #     drug_ids, NEIGHBOR_SAMPLE_SIZE, hops=2  # 可以尝试 hops=2 ，3不行会出现 MemoryError
            # )
            # 消融实验 A8_1
            # adj_entity, adj_relation, edge_weights = neighbor_sampler.sample(
            #     drug_ids, sample_size=64, hops=4, per_hop_limit=32
            # )
            # 消融实验 A8_2
            adj_entity, adj_relation, edge_weights = neighbor_sampler.sample(
                drug_ids, sample_size=64, hops=4, per_hop_limit=16
            )

            # out = model(emb, drug_entity_indices, adj_entity, adj_relation, edge_weights)
            # 按层采样，保证不同层访问不同hop领域，捕获多阶语义关系3
            out = model(drug_entity_indices=drug_entity_indices, neighbor_sampler=neighbor_sampler)
            global_embeddings.append(out.cpu())

    global_embeddings = torch.cat(global_embeddings, dim=0)
    torch.save({
        "drug_ids": dataset.valid_drug_ids,
        "global_embeddings": global_embeddings
    }, output_file)
    print(f"全局 embeddings 已保存到 {output_file}")


# ==================== 脚本入口 ====================
if __name__ == "__main__":
    embeddings_file = "models/model/drug_ablation_embeddings_A5_3.pt"  #drug_initial_embeddings.pt
    kg_file = "data/drugbank/drugbank_kg_triples_cleaned.csv"
    entity2id_file = "data/drugbank/entity2id_A5_3.txt"
    relation2id_file = "data/drugbank/relation2id_A5_3.txt"
    output_file = "models/model/drug_global_embeddings_ablation_A5_3_222.pt"

    train_global_embeddings(embeddings_file, kg_file, output_file, entity2id_file, relation2id_file)