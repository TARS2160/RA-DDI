# RA-DDI
A multimodal knowledge graph framework for drug-drug interaction prediction

## Datasets
### 公开数据集获取
drugbank：Wishart, D. S. et al. DrugBank 5.0: A Major Update to the DrugBank Database for 2018. Nucleic Acids Res. 46,
D1074–D1082 (2018).
DDInter：Xiong, G. et al. DDInter: An Online Drug-Drug Interaction Database towards Improving Clinical Decision-Making and
Patient Safety. Nucleic Acids Res. 50, D1200–D1207 (2022).
PDD Graph：Wang, M. et al. PDD Graph: Bridging Electronic Medical Records and Biomedical Knowledge Graphs via Entity Linking.
In The Semantic Web – ISWC 2017, Cham, Springer International Publishing, 219–227 (2017).

### Data Leakage Prevention and Shared Splits / 数据泄漏防范与共享划分

#### DDI-edge masking / DDI 边屏蔽

The biomedical knowledge graph used by RA-DDI was constructed exclusively from DrugBank. To prevent target DDI information from leaking through the graph structure, all triples associated with the `interacts_with` and `interaction_description` relations were removed before knowledge graph representation learning.
The masked KG was independently verified against the expected filtering result. The actual and expected masked KGs contained the same number of triples and had identical order-independent fingerprints.
Drug pairs were canonicalized as unordered pairs during leakage auditing. Therefore, `(A, B)` and `(B, A)` were treated as the same drug pair.
Before splitting each dataset, every directed drug pair was assigned a canonical unordered-pair group identifier:

```python
pair_group = "||".join(
    sorted([str(drug_i).strip(), str(drug_j).strip()])
)
```

Therefore, `(A, B)` and `(B, A)` always belong to the same `pair_group`.

The released split-generation script uses group-aware stratified splitting.
The resulting proportions are approximately 70% training, 10% validation, and 20% test. Three random seeds (`42`, `43`, and `44`) and five folds are used, resulting in 15 shared runs for each dataset.




## 文件\脚本功能描述
- `split\splits.py`: generates group-aware shared train/validation/test splits.  
  用于生成基于无向药物对分组的共享训练集、验证集和测试集划分。
