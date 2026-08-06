# RA-DDI

**RA-DDI** is a multimodal knowledge graph framework for drug–drug interaction (DDI) prediction. The framework integrates molecular representations and biomedical knowledge graph information to learn discriminative drug representations for accurate DDI prediction.

---

# Repository Structure

```text
RA-DDI/
│
├── data/                  # Datasets and preprocessing files
├── baseline/              # Baseline implementations
├── split/                 # Shared split generation scripts
├── model/                 # RA-DDI model
├── utils/                 # Utility functions
├── checkpoints/           # Trained models
├── logs/                  # Training logs
├── README.md
└── LICENSE
```

---

# Datasets

The experiments are conducted on three publicly available DDI datasets.

| Dataset | Description | Reference |
|----------|-------------|-----------|
| DrugBank | Drug–drug interaction dataset collected from DrugBank | Wishart D. S. et al., *DrugBank 5.0: A Major Update to the DrugBank Database for 2018*, Nucleic Acids Research, 2018. |
| DDInter | Clinical drug interaction database | Xiong G. et al., *DDInter: An Online Drug–Drug Interaction Database toward Improving Clinical Decision-Making and Patient Safety*, Nucleic Acids Research, 2022. |
| PDD Graph | Electronic medical record based DDI dataset | Wang M. et al., *PDD Graph: Bridging Electronic Medical Records and Biomedical Knowledge Graphs via Entity Linking*, ISWC, 2017. |

Please download the original datasets from the corresponding official sources.

---

# Data Leakage Prevention

To ensure a fair evaluation, RA-DDI adopts two complementary strategies to eliminate potential information leakage.

## DDI-edge Masking

The biomedical knowledge graph used in RA-DDI is constructed exclusively from DrugBank.

Before knowledge graph representation learning, all triples associated with

- `interacts_with`
- `interaction_description`

are removed from the knowledge graph.

This guarantees that no explicit DDI information is available during knowledge graph embedding.

The masked knowledge graph was independently verified against the expected filtering result. The actual and expected masked graphs contain exactly the same triples and produce identical order-independent fingerprints.

---

## Group-aware Shared Splits

Drug pairs are treated as **unordered pairs** during dataset splitting.

Therefore,

```
(A, B) == (B, A)
```

Before splitting, every drug pair is assigned a canonical group identifier:

```python
pair_group = "||".join(
    sorted([str(drug_i).strip(), str(drug_j).strip()])
)
```

Consequently,

```
(A, B)
(B, A)
```

always belong to the same group and will never appear in different subsets.

Group-aware stratified splitting is then applied to generate

- 70% Training
- 10% Validation
- 20% Testing

Three random seeds (`42`, `43`, `44`) and five folds are used, producing **15 shared experimental splits** for each dataset.

These shared splits are used by RA-DDI and all baseline methods to ensure a fair comparison.

---

# Baseline Models

The repository includes adapted implementations of several representative DDI prediction methods.

| Method | Official Repository |
|----------|--------------------|
| DeepDDI | https://bitbucket.org/kaistsystemsbiology/deepddi/src/master/ |
| KGNN | https://github.com/xzenglab/KGNN |
| SumGNN | https://github.com/yuanfulu/DrugKG |
| SkipGNN | https://github.com/kexinhuang12345/SkipGNN |
| DrugDAGT | https://github.com/ZhenYuanSun/DrugDAGT |
| Node2Vec + MLP | https://github.com/aditya-grover/node2vec |

Please refer to **baseline/README.md** for detailed descriptions and implementation notes.

---

# Scripts

| Script | Description |
|---------|-------------|
| `split/splits.py` | Generates group-aware shared train/validation/test splits. |
| `split/check_no_ddi.py` | Verifies that all DDI relations have been removed from the knowledge graph. |
| `split/check_no_ddi_full_audit.py` | Performs comprehensive leakage auditing between the masked knowledge graph and downstream datasets. |

---

# Reproducibility

To ensure reproducibility, all experiments

- use the same shared data splits;
- use the same evaluation metrics;
- fix the random seeds;
- prevent DDI leakage through both knowledge graph masking and group-aware splitting.

---

# Citation

If you use this repository in your research, please cite both the RA-DDI paper and the corresponding original papers of the baseline methods.
