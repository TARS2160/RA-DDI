# Baseline Models 

This repository includes implementations and adaptations of several representative drug–drug interaction (DDI) prediction methods for fair comparison with **RA-DDI**.


---

# Repository Structure 

```text
baseline/
│
├── DeepDDI/              # DeepDDI baseline
├── KGNN/                 # KGNN baseline
├── SumGNN/               # SumGNN baseline
├── SkipGNN/              # SkipGNN baseline
├── DrugDAGT/             # DrugDAGT baseline
├── Node2Vec/             # Node2Vec + MLP baseline
│
├── split/                # Shared train/validation/test splits
├── prepared/             # Preprocessed datasets
├── README.md             # Baseline description
└── LICENSE
```


---

# Baseline Descriptions 

## 1. DeepDDI

**Paper**

Ryu J., Kim H., Lee S.

*Deep Learning Improves Prediction of Drug–Drug and Drug–Food Interactions.*

Proceedings of the National Academy of Sciences (PNAS), 2018.

**Official Repository**

https://bitbucket.org/kaistsystemsbiology/deepddi/src/master/

---

## 2. KGNN

**Paper**

Lin X., et al.

Knowledge Graph Neural Network for Drug–Drug Interaction Prediction.

**Official Repository**

https://github.com/xzenglab/KGNN

---

## 3. SumGNN

**Paper**

Yu F., et al.

SumGNN: Multi-typed Drug Interaction Prediction via Knowledge Graph Summarization.

**Official Repository**

https://github.com/yuanfulu/DrugKG

---

## 4. SkipGNN

**Paper**

Huang K., et al.

SkipGNN: Predicting Molecular Interactions via Skip Graph Neural Networks.

**Official Repository**

https://github.com/kexinhuang12345/SkipGNN

---

## 5. DrugDAGT

**Paper**

DrugDAGT: Drug Interaction Prediction via Dual-Attention Graph Transformer.

**Official Repository**

https://github.com/ZhenYuanSun/DrugDAGT

---

## 6. Node2Vec + MLP

**Paper**

Grover A., Leskovec J.

node2vec: Scalable Feature Learning for Networks.

KDD, 2016.

**Official Repository**

https://github.com/aditya-grover/node2vec

---

# Notes 

### English

To ensure a fair comparison with RA-DDI, all baseline models were adapted to use the same data splits and evaluation protocol.

The modifications mainly include:

- Unified train/validation/test splits.
- Unified data preprocessing pipeline.
- Unified evaluation metrics.
- Multi-class classification support where necessary.
- Reproducible experiments with fixed random seeds.

These modifications only affect data loading and evaluation, while preserving the original model architectures.

---
