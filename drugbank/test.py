import pandas as pd
import re

# ---------------------------------
# 规则定义：根据 description 判定 label
# ---------------------------------

def classify_text(desc: str) -> str:
    text = desc.lower()

    # ---- increase 类 ----
    increase_keywords = [
        "increase", "increased", "enhance", "enhanced",
        "raise", "elevate", "boost"
    ]
    if any(k in text for k in increase_keywords):
        # 但要排除 adverse 中的 "increase risk" 情况
        if "risk" in text or "bleeding" in text or "hemorrhage" in text:
            return "adverse"
        return "increase"

    # ---- decrease 类 ----
    decrease_keywords = [
        "decrease", "decreased", "reduce", "reduced",
        "lower", "diminish", "suppress"
    ]
    if any(k in text for k in decrease_keywords):
        return "decrease"

    # ---- adverse 类 ----
    adverse_keywords = [
        "bleeding", "hemorrhage", "toxic", "toxicity",
        "risk", "adverse", "harm", "danger"
    ]
    if any(k in text for k in adverse_keywords):
        return "adverse"

    # ---- synergistic 类 ----
    synergistic_keywords = [
        "synergistic", "potentiate", "cooperate",
        "work together", "greater than individual"
    ]
    if any(k in text for k in synergistic_keywords):
        return "synergistic"

    # ---- 其它归为 interaction ----
    return "interaction"


# ---------------------------------
# 主程序
# ---------------------------------

def main():
    input_file = "drug_interactions.csv"
    output_file = "test.csv"

    df = pd.read_csv(input_file)

    # 生成 label
    df["label"] = df["description"].apply(classify_text)

    # ---- 新增：打印类别分布 ----
    print("\n样本分布（label counts）：")
    print(df["label"].value_counts())
    print("\n百分比分布（label ratio）：")
    print(df["label"].value_counts(normalize=True))

    # 保存需要的三列
    out = df[["drug1_id", "drug2_id", "label"]]
    out.to_csv(output_file, index=False)

    print("处理完成！输出文件：", output_file)


if __name__ == "__main__":
    main()
