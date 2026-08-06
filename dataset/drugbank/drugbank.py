import pandas as pd
import re

def classify_text(desc: str) -> str:
    text = desc.lower()

    # ---- increase  ----
    increase_keywords = [
        "increase", "increased", "enhance", "enhanced",
        "raise", "elevate", "boost"
    ]
    if any(k in text for k in increase_keywords):
        if "risk" in text or "bleeding" in text or "hemorrhage" in text:
            return "adverse"
        return "increase"

    # ---- decrease  ----
    decrease_keywords = [
        "decrease", "decreased", "reduce", "reduced",
        "lower", "diminish", "suppress"
    ]
    if any(k in text for k in decrease_keywords):
        return "decrease"

    # ---- adverse  ----
    adverse_keywords = [
        "bleeding", "hemorrhage", "toxic", "toxicity",
        "risk", "adverse", "harm", "danger"
    ]
    if any(k in text for k in adverse_keywords):
        return "adverse"

    # ---- synergistic  ----
    synergistic_keywords = [
        "synergistic", "potentiate", "cooperate",
        "work together", "greater than individual"
    ]
    if any(k in text for k in synergistic_keywords):
        return "synergistic"

    return "interaction"



def main():
    input_file = "drug_interactions.csv"
    output_file = "test.csv"

    df = pd.read_csv(input_file)

    df["label"] = df["description"].apply(classify_text)

    print("\n counts（label counts）：")
    print(df["label"].value_counts())
    print("\n ratio（label ratio）：")
    print(df["label"].value_counts(normalize=True))

    out = df[["drug1_id", "drug2_id", "label"]]
    out.to_csv(output_file, index=False)

    print("Finish, output", output_file)


if __name__ == "__main__":
    main()
