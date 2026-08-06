from rdflib import Graph
import pandas as pd
from itertools import combinations
import re
from collections import Counter
import os
import random

# ==== Step 1. Read TTL file ====
patients_to_diagnoses_file = "diagnose_icd_information.ttl"
patients_to_drugs_file = "drug_patients_expansion.ttl"

g_diag = Graph()
g_diag.parse(patients_to_diagnoses_file, format="turtle")

g_drug = Graph()
g_drug.parse(patients_to_drugs_file, format="turtle")

# ==== Step 2. generate patient-drug, patient-disease ====
patient_drug_rows = []
patient_disease_rows = []

for subj, pred, obj in g_drug:
    subj_id = subj.split("/")[-1]  # patient ID
    pred_str = str(pred)
    if pred_str.endswith("take_drug_name"):
        drug_name = str(obj)
        patient_drug_rows.append({"patient": subj_id, "drug": drug_name, "source": "name"})
    elif pred_str.endswith("take_drugbank_id"):
        # DrugBank ID
        match = re.search(r"drugbank:(DB\d+)", str(obj))
        if match:
            drug_id = match.group(1)
            patient_drug_rows.append({"patient": subj_id, "drug": drug_id, "source": "drugbank_id"})

for subj, pred, obj in g_diag:
    subj_id = subj.split("/")[-1]
    pred_str = str(pred)
    if pred_str.endswith("diagnoses_icd9"):
        icd_code = obj.split("/")[-1]
        patient_disease_rows.append({"patient": subj_id, "disease": icd_code})

df_patient_drug = pd.DataFrame(patient_drug_rows)
df_patient_disease = pd.DataFrame(patient_disease_rows)

# ==== Step 3.  DDI table   ====
cooccurrence = Counter()

for patient, group in df_patient_drug.groupby("patient"):
    drugs = group["drug"].unique()
    if len(drugs) > 1:
        for d1, d2 in combinations(sorted(drugs), 2):
            cooccurrence[(d1, d2)] += 1

df_cooccur = pd.DataFrame([{"drug1": d1, "drug2": d2, "weight": w}
                           for (d1, d2), w in cooccurrence.items()])

# ==== Step 4.  drug-disease  ====
df_drug_disease = pd.merge(df_patient_drug, df_patient_disease, on="patient", how="inner")
df_drug_disease = df_drug_disease[["drug", "disease"]].drop_duplicates()

# ==== Step 5. coverage  ====
total_drug_count = df_patient_drug["drug"].nunique()
mapped_drug_count = df_patient_drug[df_patient_drug["source"]=="drugbank_id"]["drug"].nunique()
coverage = mapped_drug_count / total_drug_count

print("==== report ====")
print(f"patient number: {df_patient_drug['patient'].nunique()}")
print(f"drug number: {total_drug_count}")
print(f"mapped DrugBank drugs number: {mapped_drug_count} (coverage {coverage:.2%})")
print(f"disease line number: {df_patient_disease['disease'].nunique()}")
print(f"drug-drug line number: {len(df_cooccur)}")
print(f"drug-disease line number: {len(df_drug_disease)}")

# ==== Step 6. train data  ====
print("\n==== train data   ====")
positives = df_cooccur[["drug1", "drug2"]].copy()
positives["label"] = 1

drugs = list(df_patient_drug["drug"].unique())
pos_set = set(zip(positives["drug1"], positives["drug2"]))

num_negatives = len(positives)
negatives = set()
while len(negatives) < num_negatives:
    d1, d2 = random.sample(drugs, 2)
    if (d1, d2) not in pos_set and (d2, d1) not in pos_set:
        negatives.add((d1, d2))

negatives = pd.DataFrame(list(negatives), columns=["drug1", "drug2"])
negatives["label"] = 0

ddi_df = pd.concat([positives, negatives], ignore_index=True)
ddi_df.rename(columns={"drug1": "first drug id", "drug2": "second drug id"}, inplace=True)

print(f"DDI number: {len(ddi_df)} (positives {len(positives)}, negatives {len(negatives)})")

pattern = r"^DB\d+$"
ddi_df_dbid = ddi_df[
    ddi_df["first drug id"].str.match(pattern) & ddi_df["second drug id"].str.match(pattern)
].copy()

# ==== Step 6. save  ====
os.makedirs("data/PDD_graph", exist_ok=True)
# df_patient_drug.to_csv("data/PDD_graph/patient_drug.csv", index=False)
# df_cooccur.to_csv("data/PDD_graph/drug_drug_cooccurrence.csv", index=False)
# df_drug_disease.to_csv("data/PDD_graph/drug_disease.csv", index=False)
# ddi_df.to_csv("test2.csv", index=False)
ddi_df_dbid.to_csv("pdd_graph.csv", index=False)
print("✅already save patient_drug.csv, drug_drug_cooccurrence.csv, drug_disease.csv, test2.csv")
