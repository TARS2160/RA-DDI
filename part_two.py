import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")


# ----------------------------
# 1. 数据集定义
# ----------------------------
class DrugDataset(Dataset):
    def __init__(self, smiles_tokens, text_input_ids, text_attention_masks, kg_list, pos_pairs=None, max_smiles_len=50):
        self.smiles_tokens = smiles_tokens
        self.text_input_ids = text_input_ids
        self.text_attention_masks = text_attention_masks
        self.kg_list = kg_list
        self.pos_pairs = pos_pairs if pos_pairs else []
        self.max_smiles_len = max_smiles_len

    def __len__(self):
        return len(self.smiles_tokens)

    def __getitem__(self, idx):
        # 确保SMILES序列长度一致
        smiles_tokens = self.smiles_tokens[idx]
        if len(smiles_tokens) < self.max_smiles_len:
            # 填充到固定长度
            smiles_tokens = smiles_tokens + [0] * (self.max_smiles_len - len(smiles_tokens))
        else:
            # 截断到固定长度
            smiles_tokens = smiles_tokens[:self.max_smiles_len]

        smiles = torch.tensor(smiles_tokens, dtype=torch.long)
        input_ids = torch.tensor(self.text_input_ids[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.text_attention_masks[idx], dtype=torch.long)
        kg = torch.tensor(self.kg_list[idx], dtype=torch.float)
        return smiles, input_ids, attention_mask, kg, idx


# ----------------------------
# 2. Masking函数
# ----------------------------
def mask_text_tokens(input_ids, mask_ratio=0.15, mask_token_id=103):
    input_ids = input_ids.clone()
    mask = torch.rand_like(input_ids.float()) < mask_ratio
    mask = mask & (input_ids != 0)
    input_ids[mask] = mask_token_id
    return input_ids.to(device)


def mask_smiles_tokens(smiles_ids, mask_ratio=0.15, vocab_size=200):
    smiles_ids = smiles_ids.clone()
    mask_token_id = vocab_size + 1
    mask = torch.rand_like(smiles_ids.float()) < mask_ratio
    mask = mask & (smiles_ids != 0)
    smiles_ids[mask] = mask_token_id
    return smiles_ids.to(device)


# ----------------------------
# 3. 对比学习batch构造
# ----------------------------
def create_contrastive_batch(batch_data, pos_pairs, max_augment=1):
    batch_indices = [item[4] for item in batch_data]
    batch_size = len(batch_indices)

    # 原始数据
    smiles_batch = torch.stack([item[0] for item in batch_data]).to(device)
    text_ids_batch = torch.stack([item[1] for item in batch_data]).to(device)
    attention_mask_batch = torch.stack([item[2] for item in batch_data]).to(device)
    kg_batch = torch.stack([item[3] for item in batch_data]).to(device)

    # 为正样本创建增强数据
    augmented_smiles, augmented_text_ids, augmented_attention_mask, augmented_kg = [], [], [], []

    for i, idx in enumerate(batch_indices):
        pos_indices = [j for i, j in pos_pairs if i == idx] + [i for i, j in pos_pairs if j == idx]
        batch_pos = [pos for pos in pos_indices if pos in batch_indices]

        if batch_pos:
            selected_pos = random.sample(batch_pos, min(max_augment, len(batch_pos)))
            for pos_idx in selected_pos:
                pos_in_batch = batch_indices.index(pos_idx)
                augmented_smiles.append(batch_data[pos_in_batch][0])
                augmented_text_ids.append(batch_data[pos_in_batch][1])
                augmented_attention_mask.append(batch_data[pos_in_batch][2])
                augmented_kg.append(batch_data[pos_in_batch][3])

    if augmented_smiles:
        augmented_smiles = torch.stack(augmented_smiles).to(device)
        augmented_text_ids = torch.stack(augmented_text_ids).to(device)
        augmented_attention_mask = torch.stack(augmented_attention_mask).to(device)
        augmented_kg = torch.stack(augmented_kg).to(device)

        all_smiles = torch.cat([smiles_batch, augmented_smiles], dim=0)
        all_text_ids = torch.cat([text_ids_batch, augmented_text_ids], dim=0)
        all_attention_mask = torch.cat([attention_mask_batch, augmented_attention_mask], dim=0)
        all_kg = torch.cat([kg_batch, augmented_kg], dim=0)

        return all_smiles, all_text_ids, all_attention_mask, all_kg, batch_indices, len(augmented_smiles)
    else:
        return smiles_batch, text_ids_batch, attention_mask_batch, kg_batch, batch_indices, 0


# ----------------------------
# 4. 模型定义
# ----------------------------
class SimpleDrugEncoder(nn.Module):
    def __init__(self, hidden_dim=256, pretrained_model_path=None, smiles_vocab_size=200):
        super().__init__()

        if pretrained_model_path is None:
            pretrained_model_path = "D:/study/gnn/DDKG-main/DDKG-main/models/huggingface/biobert_v1.1_pubmed"

        if not os.path.exists(pretrained_model_path):
            raise ValueError(f"预训练模型路径不存在: {pretrained_model_path}")

        print(f"加载预训练模型: {pretrained_model_path}")
        self.text_encoder = AutoModel.from_pretrained(pretrained_model_path)

        for param in list(self.text_encoder.parameters())[:-4]:
            param.requires_grad = False

        self.text_proj = nn.Linear(768, hidden_dim)

        self.smiles_embedding = nn.Embedding(smiles_vocab_size + 2, hidden_dim)
        self.smiles_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.smiles_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.kg_proj = nn.Linear(50, hidden_dim)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_ids, attention_mask, smiles_input, kg_input):
        text_ids = text_ids.to(device)
        attention_mask = attention_mask.to(device)
        smiles_input = smiles_input.to(device)
        kg_input = kg_input.to(device)

        # 文本编码
        text_outputs = self.text_encoder(input_ids=text_ids, attention_mask=attention_mask)
        text_emb = text_outputs.last_hidden_state[:, 0, :]
        text_emb = self.text_proj(text_emb)

        # SMILES编码
        smiles_emb = self.smiles_embedding(smiles_input)
        lstm_out, (hidden, _) = self.smiles_lstm(smiles_emb)
        smiles_emb = torch.cat([hidden[0], hidden[1]], dim=1)
        smiles_emb = self.smiles_proj(smiles_emb)
        smiles_emb = self.layer_norm(smiles_emb)

        # KG编码
        kg_emb = self.kg_proj(kg_input)

        return text_emb, smiles_emb, kg_emb


class SimpleCrossModalAttention(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_emb, smiles_emb, kg_emb):
        modalities = torch.stack([text_emb, smiles_emb, kg_emb], dim=1)
        attended, _ = self.attention(modalities, modalities, modalities)
        fused = attended.mean(dim=1)
        fused = self.proj(fused)
        return fused

# ----------------------------
# 5. 对比损失函数
# ----------------------------
def contrastive_loss(embeddings, original_batch_size, num_augmented):
    if embeddings.size(0) <= 1:
        return torch.tensor(0.0, device=device)

    embeddings = F.normalize(embeddings, dim=1)
    similarity_matrix = torch.mm(embeddings, embeddings.t())

    labels = torch.arange(original_batch_size, device=device)

    if num_augmented > 0:
        expanded_labels = []
        for i in range(original_batch_size):
            expanded_labels.append(i)
            for j in range(num_augmented // original_batch_size + 1):
                if len(expanded_labels) < embeddings.size(0):
                    expanded_labels.append(i)
        labels = torch.tensor(expanded_labels[:embeddings.size(0)], device=device)

    temperature = 0.1
    similarity_matrix = similarity_matrix / temperature

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


# ----------------------------
# 6. 训练循环（修复嵌入生成问题）
# ----------------------------
def optimized_training(dataset, pos_pairs, drug_ids, epochs=5, batch_size=8, lr=1e-4):
    if len(pos_pairs) > 100000:
        print(f"正样本对数量过多 ({len(pos_pairs)})，进行随机采样...")
        pos_pairs = random.sample(pos_pairs, 100000)

    pos_pairs_set = set(pos_pairs) | set((j, i) for i, j in pos_pairs)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda x: x, num_workers=0)

    model = SimpleDrugRepresentationModel().to(device)

    accumulation_steps = 2
    effective_batch_size = batch_size * accumulation_steps

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.text_encoder.parameters(), 'lr': lr * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'text_encoder' not in n], 'lr': lr}
    ], weight_decay=1e-5)

    print(f"药物数量: {len(drug_ids)}")
    print(f"正样本对数量: {len(pos_pairs)}")
    print(f"批次大小: {batch_size} (有效批次: {effective_batch_size})")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}") as tepoch:
            for step, batch_data in enumerate(tepoch):
                smiles_batch, text_ids_batch, attention_mask_batch, kg_batch, batch_indices, num_augmented = \
                    create_contrastive_batch(batch_data, pos_pairs_set, max_augment=1)

                if len(batch_indices) <= 1:
                    continue

                text_ids_batch_masked = mask_text_tokens(text_ids_batch)
                smiles_batch_masked = mask_smiles_tokens(smiles_batch)

                fused_emb = model(text_ids_batch_masked, attention_mask_batch, smiles_batch_masked, kg_batch)

                loss = contrastive_loss(fused_emb, len(batch_indices), num_augmented)
                loss = loss / accumulation_steps

                loss.backward()

                if (step + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps
                tepoch.set_postfix({
                    "loss": loss.item() * accumulation_steps,
                    "avg_loss": total_loss / (step + 1)
                })

        if len(dataloader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"checkpoint_epoch_{epoch + 1}.pt")
            print(f"检查点已保存: checkpoint_epoch_{epoch + 1}.pt")

    # 修复嵌入生成：逐个样本处理，避免长度不一致问题
    print("生成最终嵌入...")
    all_embeddings = []
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="生成嵌入"):
            # 逐个样本处理，避免batch中长度不一致的问题
            smiles, input_ids, attn_mask, kg, idx = dataset[i]

            # 添加batch维度
            smiles = smiles.unsqueeze(0).to(device)
            input_ids = input_ids.unsqueeze(0).to(device)
            attn_mask = attn_mask.unsqueeze(0).to(device)
            kg = kg.unsqueeze(0).to(device)

            try:
                embedding = model(input_ids, attn_mask, smiles, kg)
                all_embeddings.append(embedding.cpu())
            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                # 如果出错，使用零向量作为占位符
                zero_embedding = torch.zeros(1, 256)  # 假设隐藏维度是256
                all_embeddings.append(zero_embedding)

    all_embeddings = torch.cat(all_embeddings, dim=0)

    torch.save({
        "drug_ids": drug_ids,
        "embeddings": all_embeddings,
        "model_state": model.state_dict()
    }, "models/model/drug_ablation_embeddings_A5_3.pt")  #"models/model/drug_initial_embeddings.pt"

    print(f"药物初始嵌入已保存，形状: {all_embeddings.shape}")
    return all_embeddings

# ----------------------------
# 7. 开始训练
# ----------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # 1) 读取SMILES
    print("加载SMILES数据...")
    smiles_df = pd.read_csv("data/drugbank/drug_smile_filtered.csv")
    drug_ids = smiles_df["drugbank_id"].tolist()
    smiles_list = smiles_df["smiles"].tolist()

    # SMILES -> token序列
    def tokenize_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        return [atom.GetSymbol() for atom in mol.GetAtoms()]


    print("Tokenizing SMILES...")
    token2id = {}
    smiles_tokens = []
    max_smiles_len = 50  # 固定长度

    for smi in tqdm(smiles_list, desc="处理SMILES"):
        toks = tokenize_smiles(smi)
        ids = []
        for t in toks:
            if t not in token2id:
                token2id[t] = len(token2id) + 1
            ids.append(token2id[t])

        # 统一处理为固定长度
        if len(ids) < max_smiles_len:
            ids = ids + [0] * (max_smiles_len - len(ids))
        else:
            ids = ids[:max_smiles_len]
        smiles_tokens.append(ids)

    # 2) 文本tokenize
    print("处理文本数据...")
    core_df = pd.read_csv("data/drugbank/drugbank_core_data.csv")
    desc_list = []
    for did in drug_ids:
        desc = core_df.loc[core_df["drugbank_id"] == did, "description"].values
        if len(desc) > 0 and isinstance(desc[0], str):
            desc_list.append(desc[0])
        else:
            desc_list.append("")

    tokenizer = AutoTokenizer.from_pretrained("D:/study/gnn/DDKG-main/DDKG-main/models/huggingface/biobert_v1.1_pubmed")
    encodings = tokenizer(desc_list, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    text_input_ids = encodings["input_ids"]
    text_attention_masks = encodings["attention_mask"]

    # 3) 读取KG属性
    print("处理KG数据...")
    kg_df = pd.read_csv("data/drugbank/drugbank_kg_triples_cleaned.csv")
    rel_vocab = {}
    kg_list = []
    for did in tqdm(drug_ids, desc="处理KG"):
        triples = kg_df[kg_df["head"] == did]
        rels = triples["relation"].tolist()
        vec = np.zeros(50, dtype=np.float32)
        for r in rels:
            if r not in rel_vocab and len(rel_vocab) < 50:
                rel_vocab[r] = len(rel_vocab)
            if r in rel_vocab:
                vec[rel_vocab[r]] = 1.0
        kg_list.append(vec)

    # 4) 正样本对
    print("构建正样本对...")
    pos_pairs = []
    for g, group_df in core_df.groupby("groups"):
        ids_in_group = group_df["drugbank_id"].tolist()
        indices = [drug_ids.index(did) for did in ids_in_group if did in drug_ids]
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    pos_pairs.append((indices[i], indices[j]))

    print(f"最终正样本对数量: {len(pos_pairs)}")

    # 构造dataset（指定最大SMILES长度）
    dataset = DrugDataset(smiles_tokens, text_input_ids, text_attention_masks, kg_list, pos_pairs, max_smiles_len=50)

    # 开始训练
    embeddings = optimized_training(dataset, pos_pairs, drug_ids, epochs=3, batch_size=4, lr=1e-4)

    print("训练完成!")
