import pandas as pd

ddinter_file = "ddinter_downloads_code_V.csv"
drugbank_file = "dataset/drugbank/drugbank_core_data.csv"

df_ddinter = pd.read_csv(ddinter_file)
df_drugbank = pd.read_csv(drugbank_file, usecols=["drugbank_id", "name"])

df_ddinter["Drug_A"] = df_ddinter["Drug_A"].str.strip().str.lower()
df_ddinter["Drug_B"] = df_ddinter["Drug_B"].str.strip().str.lower()
df_drugbank["name"] = df_drugbank["name"].str.strip().str.lower()

name_to_dbid = dict(zip(df_drugbank["name"], df_drugbank["drugbank_id"]))

mapped_rows = []
unmapped_rows = []

for _, row in df_ddinter.iterrows():
    drug_a = row["Drug_A"]
    drug_b = row["Drug_B"]
    level = row["Level"]

    id_a = name_to_dbid.get(drug_a, row["DDInterID_A"])  
    id_b = name_to_dbid.get(drug_b, row["DDInterID_B"])

    if drug_a in name_to_dbid and drug_b in name_to_dbid:
        mapped_rows.append({"first drug id": id_a, "second drug id": id_b, "label": level})
    else:
        unmapped_rows.append({"first drug id": id_a, "second drug id": id_b, "label": level})

df_mapped = pd.DataFrame(mapped_rows)
df_unmapped = pd.DataFrame(unmapped_rows)

df_final = pd.concat([df_mapped, df_unmapped], ignore_index=True)

df_final.to_csv("DDInter.csv", index=False)

all_drugs = set(df_ddinter["Drug_A"].unique()) | set(df_ddinter["Drug_B"].unique())
mapped_drugs = {d for d in all_drugs if d in name_to_dbid}

coverage = len(mapped_drugs) / len(all_drugs)

print("==== Results ====")
print(f"Total drugs: {len(all_drugs)}")
print(f"Mapped to DrugBank: {len(mapped_drugs)}")
print(f"Coverage: {coverage:.2%}")
print(f"output: DDInter.csv")
