from rdflib import Graph
import pandas as pd
from itertools import combinations
import re
from collections import Counter
import os
import random

# ==== Step 1. 读取 TTL 文件 ====
patients_to_diagnoses_file = "diagnose_icd_information.ttl"
patients_to_drugs_file = "drug_patients_expansion.ttl"

g_diag = Graph()
g_diag.parse(patients_to_diagnoses_file, format="turtle")

g_drug = Graph()
g_drug.parse(patients_to_drugs_file, format="turtle")

# ==== Step 2. 解析 RDF 三元组，生成 patient-drug, patient-disease ====
patient_drug_rows = []
patient_disease_rows = []

# 处理药物
for subj, pred, obj in g_drug:
    subj_id = subj.split("/")[-1]  # patient ID
    pred_str = str(pred)
    if pred_str.endswith("take_drug_name"):
        drug_name = str(obj)
        patient_drug_rows.append({"patient": subj_id, "drug": drug_name, "source": "name"})
    elif pred_str.endswith("take_drugbank_id"):
        # 提取 DrugBank ID
        match = re.search(r"drugbank:(DB\d+)", str(obj))
        if match:
            drug_id = match.group(1)
            patient_drug_rows.append({"patient": subj_id, "drug": drug_id, "source": "drugbank_id"})

# 处理疾病
for subj, pred, obj in g_diag:
    subj_id = subj.split("/")[-1]
    pred_str = str(pred)
    if pred_str.endswith("diagnoses_icd9"):
        icd_code = obj.split("/")[-1]
        patient_disease_rows.append({"patient": subj_id, "disease": icd_code})

df_patient_drug = pd.DataFrame(patient_drug_rows)
df_patient_disease = pd.DataFrame(patient_disease_rows)

# ==== Step 3. 生成 drug-drug 共现表 ====
cooccurrence = Counter()

for patient, group in df_patient_drug.groupby("patient"):
    drugs = group["drug"].unique()
    if len(drugs) > 1:
        for d1, d2 in combinations(sorted(drugs), 2):
            cooccurrence[(d1, d2)] += 1

df_cooccur = pd.DataFrame([{"drug1": d1, "drug2": d2, "weight": w}
                           for (d1, d2), w in cooccurrence.items()])

# ==== Step 4. 生成 drug-disease 表 ====
df_drug_disease = pd.merge(df_patient_drug, df_patient_disease, on="patient", how="inner")
df_drug_disease = df_drug_disease[["drug", "disease"]].drop_duplicates()

# ==== Step 5. 映射覆盖率与统计 ====
total_drug_count = df_patient_drug["drug"].nunique()
mapped_drug_count = df_patient_drug[df_patient_drug["source"]=="drugbank_id"]["drug"].nunique()
coverage = mapped_drug_count / total_drug_count

print("==== 数据统计报告 ====")
print(f"患者数: {df_patient_drug['patient'].nunique()}")
print(f"药物总数: {total_drug_count}")
print(f"映射到DrugBank的药物数: {mapped_drug_count} (覆盖率 {coverage:.2%})")
print(f"疾病总数: {df_patient_disease['disease'].nunique()}")
print(f"药物-药物对数量: {len(df_cooccur)}")
print(f"药物-疾病对数量: {len(df_drug_disease)}")

# ==== Step 6. 构造 DDI 训练数据（正负样本）====
print("\n==== 构造 DDI 训练数据 ====")
positives = df_cooccur[["drug1", "drug2"]].copy()
positives["label"] = 1

# 药物全集
drugs = list(df_patient_drug["drug"].unique())
pos_set = set(zip(positives["drug1"], positives["drug2"]))

# 采样负样本（比例 1:1）
num_negatives = len(positives)
negatives = set()
while len(negatives) < num_negatives:
    d1, d2 = random.sample(drugs, 2)
    if (d1, d2) not in pos_set and (d2, d1) not in pos_set:
        negatives.add((d1, d2))

negatives = pd.DataFrame(list(negatives), columns=["drug1", "drug2"])
negatives["label"] = 0

# 合并正负样本
ddi_df = pd.concat([positives, negatives], ignore_index=True)
ddi_df.rename(columns={"drug1": "first drug id", "drug2": "second drug id"}, inplace=True)

print(f"最终 DDI 数据大小: {len(ddi_df)} (正样本 {len(positives)}, 负样本 {len(negatives)})")

# ==== Step 6.1. 过滤：仅保留DB ID ====
pattern = r"^DB\d+$"
ddi_df_dbid = ddi_df[
    ddi_df["first drug id"].str.match(pattern) & ddi_df["second drug id"].str.match(pattern)
].copy()

# ==== Step 7. 保存 ====
os.makedirs("data/PDD_graph", exist_ok=True)
# df_patient_drug.to_csv("data/PDD_graph/patient_drug.csv", index=False)
# df_cooccur.to_csv("data/PDD_graph/drug_drug_cooccurrence.csv", index=False)
# df_drug_disease.to_csv("data/PDD_graph/drug_disease.csv", index=False)
# ddi_df.to_csv("test2.csv", index=False)
ddi_df_dbid.to_csv("test2.csv", index=False)
print("✅ 已保存 patient_drug.csv, drug_drug_cooccurrence.csv, drug_disease.csv, test2.csv")
