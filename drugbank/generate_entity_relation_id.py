import pandas as pd
import torch

def generate_entity_relation_with_embeddings(kg_csv, embedding_file, entity_out, relation_out, output_embedding_file):
    # 1. 读取药物初始 embeddings
    data = torch.load(embedding_file)
    drug_ids = list(data["drug_ids"])  # 保证顺序
    drug_emb = data["embeddings"]
    embed_dim = drug_emb.size(1)

    # 2. 读取 KG
    df = pd.read_csv(kg_csv)

    # 3. 构造实体集合
    kg_entities = set(df["head"]).union(set(df["tail"]))
    relations = sorted(set(df["relation"]))

    # 只保留 KG 中出现的实体
    # 保证药物 DBID 排在前面
    entities = drug_ids + [e for e in kg_entities if e not in drug_ids]

    # 4. 生成映射
    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    relation2id = {rel: idx for idx, rel in enumerate(relations)}

    # 5. 保存映射
    with open(entity_out, "w", encoding="utf-8") as f:
        for entity, idx in entity2id.items():
            f.write(f"{entity}\t{idx}\n")

    with open(relation_out, "w", encoding="utf-8") as f:
        for rel, idx in relation2id.items():
            f.write(f"{rel}\t{idx}\n")

    print("=== 统计结果 ===")
    print(f"实体总数: {len(entities)} (药物 {len(drug_ids)} + 非药物 {len(entities) - len(drug_ids)})")
    print(f"关系总数: {len(relations)}")

    # 6. 初始化全量 embedding
    all_embeddings = torch.zeros((len(entities), embed_dim))
    # 药物 embedding
    all_embeddings[:len(drug_ids)] = drug_emb
    # 其他实体随机初始化
    all_embeddings[len(drug_ids):] = torch.randn(len(entities) - len(drug_ids), embed_dim)

    # 保存
    torch.save({"entity2id": entity2id, "relation2id": relation2id, "embeddings": all_embeddings}, output_embedding_file)
    print(f"完整实体 embedding 已保存到 {output_embedding_file}")

    return entity2id, relation2id


# 用法
entity2id, relation2id = generate_entity_relation_with_embeddings(
    "drugbank_kg_triples_cleaned.csv",
    "model/drug_initial_embeddings.pt",
    "entity2id.txt",
    "relation2id.txt",
    "model/entity_embeddings.pt"
)

# entity2id.txt
# relation2id.txt
# models/model/drug_initial_embeddings.pt
