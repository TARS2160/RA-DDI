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


# ==================== 邻居采样 ====================
class NeighborSampler:
    def __init__(self, neighbor_dict, drug_id_to_entity_idx):
        self.neighbor_dict = neighbor_dict
        self.drug_id_to_entity_idx = drug_id_to_entity_idx

    def sample(self, drug_ids, sample_size):
        adj_entity = []
        adj_relation = []
        edge_weights = []

        for drug_id in drug_ids:
            # 使用entity2id中的索引
            entity_idx = self.drug_id_to_entity_idx.get(drug_id)
            if entity_idx is None:
                # 如果药物不在entity2id中，使用全0邻居
                adj_entity.append([0] * sample_size)
                adj_relation.append([0] * sample_size)
                edge_weights.append([0.0] * sample_size)
                continue

            neighbors = self.neighbor_dict.get(entity_idx, [])

            # 确保邻居列表长度一致
            if len(neighbors) == 0:
                neighbors = [(0, entity_idx)]  # 自环

            # 采样或填充到指定大小
            if len(neighbors) >= sample_size:
                sampled_neighbors = random.sample(neighbors, sample_size)
            else:
                # 如果邻居不够，重复采样
                sampled_neighbors = neighbors
                while len(sampled_neighbors) < sample_size:
                    remaining = sample_size - len(sampled_neighbors)
                    sampled_neighbors.extend(random.choices(neighbors, k=min(remaining, len(neighbors))))

            adj_entity.append([n[1] for n in sampled_neighbors])
            adj_relation.append([n[0] for n in sampled_neighbors])
            edge_weights.append([1.0] * len(sampled_neighbors))

        # 验证所有列表长度一致
        assert all(len(lst) == sample_size for lst in adj_entity), "adj_entity长度不一致"
        assert all(len(lst) == sample_size for lst in adj_relation), "adj_relation长度不一致"
        assert all(len(lst) == sample_size for lst in edge_weights), "edge_weights长度不一致"

        return (torch.tensor(adj_entity, dtype=torch.long, device=DEVICE),
                torch.tensor(adj_relation, dtype=torch.long, device=DEVICE),
                torch.tensor(edge_weights, dtype=torch.float, device=DEVICE))


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

class RelationalAttentionRGCN(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim, num_layers):
        super(RelationalAttentionRGCN, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        # 实体嵌入层，用于处理KG中的所有实体
        self.entity_embedding = nn.Embedding(num_entities, embed_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight)

        self.rgcn_layers = nn.ModuleList([
            RGCNAggregator(embed_dim, num_relations) for _ in range(num_layers)
        ])
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4)

    def forward(self, drug_emb, drug_entity_indices, adj_entity, adj_relation, edge_weights):
        # 将初始嵌入与KG实体嵌入结合
        entity_emb = self.entity_embedding(drug_entity_indices)
        combined_emb = drug_emb + entity_emb  # 残差连接

        x = combined_emb.unsqueeze(0)
        for layer in self.rgcn_layers:
            x = layer(x, adj_entity, adj_relation, edge_weights)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.squeeze(0)

#==================== 模型 ====================
class RelationalAttentionRGCN(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim, num_layers):
        super(RelationalAttentionRGCN, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        # 实体嵌入层，用于处理KG中的所有实体
        self.entity_embedding = nn.Embedding(num_entities, embed_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight)

        self.rgcn_layers = nn.ModuleList([
            RGCNAggregator(embed_dim, num_relations) for _ in range(num_layers)
        ])

        # 多头注意力：我们将对每层的输出做 attention 再融合
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4)

    def forward(self,
                drug_emb=None,
                drug_entity_indices=None,
                adj_entity=None,
                adj_relation=None,
                edge_weights=None,
                neighbor_sampler=None):

        # --- 准备 combined_emb ---
        # 如果 drug_emb 提供，直接使用它（通常是float tensor）
        if drug_emb is not None:
            combined_emb = drug_emb  # [B, D], float
        else:
            # 否则从 entity_embedding 查表（需要 drug_entity_indices）
            if drug_entity_indices is None:
                raise ValueError("Either drug_emb or drug_entity_indices must be provided.")
            if drug_entity_indices.dtype != torch.long:
                drug_entity_indices = drug_entity_indices.long()
            device = next(self.entity_embedding.parameters()).device
            drug_entity_indices = drug_entity_indices.to(device)
            combined_emb = self.entity_embedding(drug_entity_indices)  # [B, D]

        # 保证 tensor 在正确设备
        combined_emb = combined_emb.to(next(self.entity_embedding.parameters()).device)

        # 变换为 attention 要求的 (seq_len, batch, dim) 格式，这里先把 node emb 做成 [1, B, D]
        x = combined_emb.unsqueeze(0)  # [1, B, D]

        # --- 遍历每层 RGCN，支持按层动态采样或使用外部邻接 ---
        layer_outputs = []
        for l, layer in enumerate(self.rgcn_layers):
            if neighbor_sampler is not None:
                sampler_input = drug_entity_indices if drug_entity_indices is not None else None
                adj_entity_l, adj_relation_l, edge_weights_l = neighbor_sampler.sample(
                    sampler_input,
                    sample_size=64,
                    hops=l + 1,
                    per_hop_limit=16
                )
            else:
                # 使用外部传入的邻接（相同邻接用于每层）
                if adj_entity is None or adj_relation is None or edge_weights is None:
                    raise ValueError("adj_entity/adj_relation/edge_weights must be provided if neighbor_sampler is None.")
                # 保证在正确的 device
                adj_entity_l = adj_entity.to(next(self.entity_embedding.parameters()).device)
                adj_relation_l = adj_relation.to(next(self.entity_embedding.parameters()).device)
                edge_weights_l = edge_weights.to(next(self.entity_embedding.parameters()).device)

            # 调用 RGCN 层聚合；layer 期望 node_emb 形状 [1, B, D]
            x = layer(x, adj_entity_l, adj_relation_l, edge_weights_l)  # 返回 [1, B, D]
            layer_outputs.append(x)

        # --- 使用 MultiheadAttention 融合各层输出 ---
        # 将多个层的输出堆叠为 seq_len = num_layers
        attn_input = torch.cat(layer_outputs, dim=0)  # [num_layers, B, D]
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        # 对层维度做平均融合为最终 embedding
        fused = attn_output.mean(dim=0)  # [B, D]

        return fused  

class RGCNAggregator(nn.Module):
    def __init__(self, embed_dim, num_relations):
        super(RGCNAggregator, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_relations, embed_dim, embed_dim))
        self.residual = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.W)

        # 在RGCNAggregator中也添加entity_embedding
        self.entity_embedding = None

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
    # 加载所有实体映射
    entity2id = load_mapping(entity2id_file, limit=None)
    relation2id = load_mapping(relation2id_file)

    num_entities = len(entity2id)
    num_relations = len(relation2id)
    print(f"实体数量: {num_entities}, 关系数量: {num_relations}")

    dataset = DrugEmbeddingDataset(embeddings_file, kg_file, entity2id)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    neighbor_dict = build_neighbor_dict(dataset.kg_triples, entity2id, relation2id)
    neighbor_sampler = NeighborSampler(neighbor_dict, dataset.drug_id_to_entity_idx)
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

            adj_entity, adj_relation, edge_weights = neighbor_sampler.sample(drug_ids, NEIGHBOR_SAMPLE_SIZE)
            
            # 按层采样，保证不同层访问不同hop领域，捕获多阶语义关系3
            out = model(drug_entity_indices=drug_entity_indices, neighbor_sampler=neighbor_sampler)
            global_embeddings.append(out.cpu())

    global_embeddings = torch.cat(global_embeddings, dim=0)
    torch.save({
        "drug_ids": dataset.valid_drug_ids,
        "global_embeddings": global_embeddings
    }, output_file)
    print(f"全局 embeddings 已保存到 {output_file}")


# ==================== 开始训练 ====================
if __name__ == "__main__":
    embeddings_file = "model/drug_initial_embeddings.pt"  #drug_initial_embeddings.pt
    kg_file = "drugbank/drugbank_kg_triples_cleaned.csv"
    entity2id_file = "drugbank/entity2id.txt"
    relation2id_file = "drugbank/relation2id.txt"
    output_file = "model/drug_global_embeddings.pt"

    train_global_embeddings(embeddings_file, kg_file, output_file, entity2id_file, relation2id_file)
