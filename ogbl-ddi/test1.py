import pandas as pd

# 读取文件
test_df = pd.read_csv("test0.csv")
drug_smiles_df = pd.read_csv("D:/study/gnn/DDKG-main/DDKG-main/data/drugbank/drug_smile_filtered.csv")

# 获取 drug_smiles_filtered 中的所有药物ID
valid_drug_ids = set(drug_smiles_df['drugbank_id'])  # 假设列名为 drug_id

# 筛选 test0.csv 中两种药物都在 valid_drug_ids 的记录
filtered_df = test_df[
    (test_df['first drug id'].isin(valid_drug_ids)) &
    (test_df['second drug id'].isin(valid_drug_ids))
]

# 保存结果到 test1.csv
filtered_df.to_csv("test1.csv", index=False)

print(f"筛选完成！原始数据 {len(test_df)} 条，筛选后 {len(filtered_df)} 条，已保存到 test1.csv")
