# RA-DDI

A multimodal knowledge graph framework for drug–drug interaction prediction.

> **Notice**

---

## 1. Overview

RA-DDI is designed for drug–drug interaction (DDI) prediction by integrating drug representations with biomedical knowledge graph information.

The current repository contains:

- dataset preprocessing scripts;
- knowledge graph construction and cleaning scripts;
- initial and global drug representation learning scripts;
- DDI prediction scripts;
- shared data split generation scripts;
- adapted baseline implementations.

The main RA-DDI workflow is currently implemented in the following three Python scripts:

- `initial_embedding.py`
- `global_embedding.py`
- `drug_drug_interaction.py`

Detailed methodological descriptions will be added after the corresponding paper is published.

---

## 2. Repository Structure

```text
RA-DDI/
│
├── baseline/
│   └── readme.md
│
├── dataset/
│   ├── DDInter/
│   │   └── DDInter.py
│   │
│   ├── PDD Graph/
│   │   ├── change.py
│   │   ├── expansion.py
│   │   └── pdd_graph.py
│   │
│   ├── drugbank/
│   │   ├── clean.py
│   │   ├── drugbank.py
│   │   └── generate_entity_relation_id.py
│   │
│   └── split.py
│
├── drug_drug_interaction.py
├── global_embedding.py
├── initial_embedding.py
├── requirements.txt
└── README.md
```

---

## 3. File and Directory Descriptions

### 3.1 `baseline/`

This directory contains the baseline methods used for comparison with RA-DDI.

```text
baseline/
└── readme.md
```

The baseline README provides information about:

- the baseline methods used in the experiments;
- the original papers;
- the official source-code repositories;
- the adaptations made for shared data splits and unified evaluation.

The baseline implementations are adapted to use the same dataset splits and evaluation protocol as RA-DDI.

---

### 3.2 `dataset/`

This directory contains scripts for dataset preprocessing, knowledge graph construction, identifier conversion, and shared data splitting.

#### `dataset/DDInter/`

```text
dataset/DDInter/
└── DDInter.py
```

- `DDInter.py`: preprocesses the DDInter dataset and converts the original data into the format required by the subsequent experimental pipeline.

#### `dataset/PDD Graph/`

```text
dataset/PDD Graph/
├── change.py
├── expansion.py
└── pdd_graph.py
```

- `pdd_graph.py`: performs the main preprocessing operations for the PDD Graph dataset.
- `change.py`: converts intermediate PDD Graph data formats.
- `expansion.py`: performs data preprocessing for the PDD Graph dataset.

#### `dataset/drugbank/`

```text
dataset/drugbank/
├── clean.py
├── drugbank.py
└── generate_entity_relation_id.py
```

- `drugbank.py`: processes the original DrugBank data and constructs the required dataset files.
- `clean.py`: cleans the extracted DrugBank records.
- `generate_entity_relation_id.py`: generates numerical identifiers for entities and relations in the biomedical knowledge graph.

#### `dataset/split.py`

Generates the shared training, validation, and test splits used by RA-DDI and the baseline models.

Drug pairs are treated as unordered pairs during dataset splitting. Before splitting, every drug pair is assigned a canonical group identifier:

```python
pair_group = "||".join(
    sorted([str(drug_i).strip(), str(drug_j).strip()])
)
```

Therefore, `(A, B)` and `(B, A)` always belong to the same group and cannot be assigned to different subsets.

The split-generation procedure uses group-aware stratified splitting with approximate proportions of:

- 70% training;
- 10% validation;
- 20% testing.

Three random seeds (`42`, `43`, and `44`) and five folds are used, producing 15 shared splits for each dataset.

---

### 3.3 `initial_embedding.py`

This script generates the initial representations required by the subsequent RA-DDI workflow.

Its main responsibilities include preparing the initial feature representations of drugs and other biomedical entities before global knowledge graph representation learning.

---

### 3.4 `global_embedding.py`

This script learns global representations from the processed biomedical knowledge graph.

The generated global embeddings are used as one of the representation sources for downstream DDI prediction.

---

### 3.5 `drug_drug_interaction.py`

This script implements the main DDI prediction workflow.

It loads the prepared drug representations and dataset splits, performs model training and evaluation, and outputs the DDI prediction results.

---

## 4. Datasets

The experiments use the following publicly available data resources.

### 4.1 DrugBank

DrugBank provides detailed information on drugs, drug targets, drug interactions, enzymes, transporters, and related biomedical entities.

**Reference**

Wishart, D. S. et al. DrugBank 5.0: A Major Update to the DrugBank Database for 2018. *Nucleic Acids Research*, 46, D1074–D1082, 2018.

Access to the complete DrugBank dataset may require registration and acceptance of the corresponding license agreement.

---

### 4.2 DDInter

DDInter is a publicly available drug–drug interaction database designed to support clinical decision-making and patient safety.

**Reference**

Xiong, G. et al. DDInter: An Online Drug–Drug Interaction Database towards Improving Clinical Decision-Making and Patient Safety. *Nucleic Acids Research*, 50, D1200–D1207, 2022.

---

### 4.3 PDD Graph

PDD Graph connects electronic medical records with biomedical knowledge graphs through entity linking.

**Reference**

Wang, M. et al. PDD Graph: Bridging Electronic Medical Records and Biomedical Knowledge Graphs via Entity Linking. In *The Semantic Web – ISWC 2017*, Springer International Publishing, Cham, pp. 219–227, 2017.

---

## 5. Data Leakage Prevention and Shared Splits

### 5.1 DDI-Edge Masking

The biomedical knowledge graph used by RA-DDI was constructed from DrugBank.

To prevent target DDI information from leaking through the graph structure, all triples associated with the following relations were removed before knowledge graph representation learning:

```text
interacts_with
interaction_description
```

The masked knowledge graph was independently compared with the expected filtering result. The actual and expected masked knowledge graphs contained the same number of triples and produced identical order-independent fingerprints.

During leakage auditing, drug pairs were canonicalized as unordered pairs. Therefore, `(A, B)` and `(B, A)` were treated as the same drug pair.

---

### 5.2 Group-Aware Shared Splitting

All datasets are divided using group-aware stratified splitting.

The grouping mechanism ensures that:

```text
(A, B) == (B, A)
```

Consequently, reversed drug pairs cannot appear in different training, validation, or test subsets.

The same shared splits are used by RA-DDI and all compatible baseline methods to improve the fairness and reproducibility of model comparison.

---

## 6. Baseline Models

The project includes adapted implementations of representative DDI prediction methods, including:

- DeepDDI;
- Node2Vec + MLP;
- KGNN;
- SkipGNN;
- SumGNN;
- DrugDAGT.

Additional information about the original publications, official repositories, and implementation adaptations is available in:

```text
baseline/readme.md
```

Some original baseline methods were designed for different task settings, label spaces, or data formats. Necessary adaptations were therefore introduced to support:

- the shared train/validation/test splits;
- unified dataset interfaces;
- unified evaluation metrics;
- repeated experiments under fixed random seeds.

The original model architecture is preserved whenever possible.

---

## 7. Environment Setup

### 7.1 Create a Python Environment

A separate Python environment is recommended.

Using Conda:

```bash
conda create -n ra-ddi python=3.10
conda activate ra-ddi
```

The Python version should be adjusted to match the version used to generate `requirements.txt`.

---

### 7.2 Install Dependencies

After downloading the repository, install the required packages with:

```bash
python -m pip install -r requirements.txt
```

For GPU execution, ensure that the installed versions of PyTorch, DGL, CUDA, and the NVIDIA driver are mutually compatible.

---

### 7.3 Verify the Python Interpreter

Before running the scripts, verify that the expected Python interpreter is active.

On Windows:

```bash
where python
python --version
python -m pip --version
```

On Linux:

```bash
which python
python --version
python -m pip --version
```

---

## 8. Usage

The general workflow is:

```text
Dataset preparation
        ↓
Initial representation generation
        ↓
Global knowledge graph representation learning
        ↓
Drug–drug interaction prediction
```

The corresponding main scripts are:

```bash
python initial_embedding.py
python global_embedding.py
python drug_drug_interaction.py
```

The exact execution order, required paths, command-line arguments, and recommended hyperparameters depend on the local dataset organization.

---

## 9. Method

The detailed RA-DDI methodology is temporarily omitted because the corresponding manuscript has not yet been published.

This section will be updated with:

- the overall model architecture;
- multimodal input construction;
- knowledge graph representation learning;
- feature aggregation mechanisms;
- attention mechanisms;
- training objectives;
- loss functions;
- inference procedures;
- computational complexity.

---

## 10. Reproducibility

The released experimental pipeline is designed to improve reproducibility through:

- fixed random seeds;
- shared dataset splits;
- group-aware splitting of unordered drug pairs;
- DDI-edge masking in the biomedical knowledge graph;
- unified evaluation metrics;
- repeated experiments across multiple seeds and folds.

Users should record the following information when reproducing the experiments:

- operating system;
- Python version;
- CUDA version;
- PyTorch version;
- DGL version;
- GPU model;
- package versions;
- random seed;
- dataset version.

---
