import pandas as pd
import torch

def generate_entity_relation_with_embeddings(kg_csv, embedding_file, entity_out, relation_out, output_embedding_file):
    data = torch.load(embedding_file)
    drug_ids = list(data["drug_ids"])  
    drug_emb = data["embeddings"]
    embed_dim = drug_emb.size(1)

    df = pd.read_csv(kg_csv)

    kg_entities = set(df["head"]).union(set(df["tail"]))
    relations = sorted(set(df["relation"]))

    entities = drug_ids + [e for e in kg_entities if e not in drug_ids]

    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    relation2id = {rel: idx for idx, rel in enumerate(relations)}

    with open(entity_out, "w", encoding="utf-8") as f:
        for entity, idx in entity2id.items():
            f.write(f"{entity}\t{idx}\n")

    with open(relation_out, "w", encoding="utf-8") as f:
        for rel, idx in relation2id.items():
            f.write(f"{rel}\t{idx}\n")

    print("=== result ===")
    print(f"entity: {len(entities)} (drug {len(drug_ids)} + no_drug {len(entities) - len(drug_ids)})")
    print(f"relation: {len(relations)}")

    all_embeddings = torch.zeros((len(entities), embed_dim))
    all_embeddings[:len(drug_ids)] = drug_emb
    all_embeddings[len(drug_ids):] = torch.randn(len(entities) - len(drug_ids), embed_dim)

    torch.save({"entity2id": entity2id, "relation2id": relation2id, "embeddings": all_embeddings}, output_embedding_file)
    print(f"embedding save to {output_embedding_file}")

    return entity2id, relation2id


entity2id, relation2id = generate_entity_relation_with_embeddings(
    "drugbank_kg_triples_no_ddi.csv",
    "models/model/drug_ablation_embeddings_no_ddi.pt",
    "entity2id_no_ddi.txt",
    "relation2id_no_ddi.txt",
    "models/model/entity_embeddings_no_ddi.pt"
)


