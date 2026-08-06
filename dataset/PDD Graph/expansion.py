import pandas as pd
import re
from urllib.parse import quote

# data clean part
def process_nt_file(nt_file_path, csv_file_path, output_file_path):
    try:
        drugbank_df = pd.read_csv(csv_file_path)
        print(f"read the file(CSV), {len(drugbank_df)} lines")
    except Exception as e:
        print(f"error reading: {e}")
        return

    name_to_id_map = {}
    for _, row in drugbank_df.iterrows():
        common_name = str(row['Common name']).strip()
        drugbank_id = str(row['DrugBank ID']).strip()

        if common_name and common_name != 'nan' and drugbank_id:
            name_to_id_map[common_name.lower()] = drugbank_id

    try:
        with open(nt_file_path, 'r', encoding='utf-8') as f:
            nt_lines = f.readlines()
        print(f"read the file(NT), {len(nt_lines)} lines")
    except Exception as e:
        print(f"error reading: {e}")
        return

    processed_lines = []
    drug_name_pattern = r'<http://kmap\.xjtudlc\.com/pdd_data/property/take_drug_name>\s+"([^"]+)"'
    drugbank_id_pattern = r'<http://kmap\.xjtudlc\.com/pdd_data/property/take_drugbank_id>\s+<http://bio2rdf\.org/drugbank:([^>]+)>'

    for line in nt_lines:
        line = line.strip()
        if not line:
            continue

        drugbank_id_match = re.search(drugbank_id_pattern, line)
        if drugbank_id_match:
            processed_lines.append(line)
            continue

        drug_name_match = re.search(drug_name_pattern, line)
        if drug_name_match:
            drug_name = drug_name_match.group(1)
            resource_uri = line.split()[0]  

            matched_id = None
            drug_name_lower = drug_name.lower()

            if drug_name_lower in name_to_id_map:
                matched_id = name_to_id_map[drug_name_lower]
            else:
                for common_name, db_id in name_to_id_map.items():
                    if drug_name_lower in common_name or common_name in drug_name_lower:
                        matched_id = db_id
                        break

            if matched_id:
                new_triple = f"{resource_uri} <http://kmap.xjtudlc.com/pdd_data/property/take_drugbank_id> <http://bio2rdf.org/drugbank:{matched_id}> ."
                processed_lines.append(new_triple)
                print(f"mapped successfully: '{drug_name}' -> {matched_id}")
            else:
                print(f"can't find: '{drug_name}'")
                processed_lines.append(line)
        else:
            processed_lines.append(line)

    # 写入输出文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in processed_lines:
                f.write(line + '\n')
        print(f"Finished, save to: {output_file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    nt_file_path = "drug_patients.nt"  
    csv_file_path = "./drugbank/drugbank_vocabulary.csv"  
    output_file_path = "drug_patients_expansion.nt" 

    process_nt_file(nt_file_path, csv_file_path, output_file_path)
