import pandas as pd

# ==== Step 1. 读取输入文件 ====
ddinter_file = "ddinter_downloads_code_V.csv"
drugbank_file = "D:/study/gnn/DDKG-main/DDKG-main/data/drugbank/drugbank_core_data.csv"

df_ddinter = pd.read_csv(ddinter_file)
df_drugbank = pd.read_csv(drugbank_file, usecols=["drugbank_id", "name"])

# 标准化药物名称，避免大小写/空格问题
df_ddinter["Drug_A"] = df_ddinter["Drug_A"].str.strip().str.lower()
df_ddinter["Drug_B"] = df_ddinter["Drug_B"].str.strip().str.lower()
df_drugbank["name"] = df_drugbank["name"].str.strip().str.lower()

# ==== Step 2. 建立映射表 ====
name_to_dbid = dict(zip(df_drugbank["name"], df_drugbank["drugbank_id"]))

# ==== Step 3. 映射 DDInter ====
mapped_rows = []
unmapped_rows = []

for _, row in df_ddinter.iterrows():
    drug_a = row["Drug_A"]
    drug_b = row["Drug_B"]
    level = row["Level"]

    # 尝试映射
    id_a = name_to_dbid.get(drug_a, row["DDInterID_A"])  # 映射不到 → 保留原 ID
    id_b = name_to_dbid.get(drug_b, row["DDInterID_B"])

    # 判断是否完全映射
    if drug_a in name_to_dbid and drug_b in name_to_dbid:
        mapped_rows.append({"first drug id": id_a, "second drug id": id_b, "label": level})
    else:
        unmapped_rows.append({"first drug id": id_a, "second drug id": id_b, "label": level})

# ==== Step 4. 输出结果 ====
df_mapped = pd.DataFrame(mapped_rows)
df_unmapped = pd.DataFrame(unmapped_rows)

df_final = pd.concat([df_mapped, df_unmapped], ignore_index=True)

df_final.to_csv("test3.csv", index=False)

# ==== Step 5. 覆盖率计算 ====
all_drugs = set(df_ddinter["Drug_A"].unique()) | set(df_ddinter["Drug_B"].unique())
mapped_drugs = {d for d in all_drugs if d in name_to_dbid}

coverage = len(mapped_drugs) / len(all_drugs)

print("==== 映射结果 ====")
print(f"总药物数: {len(all_drugs)}")
print(f"映射到 DrugBank 的药物数: {len(mapped_drugs)}")
print(f"覆盖率: {coverage:.2%}")
print(f"输出文件: test3.csv")
