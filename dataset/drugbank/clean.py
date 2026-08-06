import pandas as pd

input_file_path = 'drugbank_kg_triples.csv'  
output_file_path = 'drugbank_kg_triples_cleaned.csv' 
truncate_length = 50  

def clean_and_truncate_id(original_id):
    if isinstance(original_id, str):
        cleaned_id = original_id.strip()[:truncate_length]
        return cleaned_id
    else:
        return str(original_id).strip()[:truncate_length]

print("begin...")

try:
    df = pd.read_csv(input_file_path)
    print(f"raw data line: {len(df)}")

    df_cleaned = df.dropna(subset=['head', 'tail', 'relation'])
    print(f"after fliter: {len(df_cleaned)}")

    df_cleaned['head'] = df_cleaned['head'].apply(clean_and_truncate_id)
    df_cleaned['tail'] = df_cleaned['tail'].apply(clean_and_truncate_id)

    before_drop = len(df_cleaned)
    df_cleaned = df_cleaned[(df_cleaned['head'] != '') & (df_cleaned['tail'] != '')]
    after_drop = len(df_cleaned)
    if before_drop != after_drop:
        print(f"warninng: remove {before_drop - after_drop} lines record.")


    df_cleaned.to_csv(output_file_path, index=False)
    print(f"finish and save to: {output_file_path}")
    print(f"final lines: {len(df_cleaned)}")

    print("\n preview:")
    print(df_cleaned.head())

except FileNotFoundError:
    print(f"error：can't find '{input_file_path}'.")
except KeyError as e:
    print(f"error：no {e} in csv. Make sure the file contain 'head', 'tail' and 'relation'.")
except Exception as e:
    print(f"unknown error: {e}")
