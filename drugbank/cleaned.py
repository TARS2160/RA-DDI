import pandas as pd

# 配置参数
input_file_path = 'drugbank_kg_triples.csv'  # 原始文件路径
output_file_path = 'drugbank_kg_triples_cleaned.csv' # 清洗后输出路径
truncate_length = 50  # 截断长度，保留前50个字符

def clean_and_truncate_id(original_id):
    """
    清洗和截断ID的函数。
    如果原始ID是字符串且长度超过限制，则截断；
    否则，保持不变。
    """
    if isinstance(original_id, str):
        # 移除首尾空格，然后截断
        cleaned_id = original_id.strip()[:truncate_length]
        return cleaned_id
    else:
        # 如果id不是字符串（例如数字），则转换为字符串后再处理
        return str(original_id).strip()[:truncate_length]

print("开始清洗数据...")

try:
    # 1. 读取原始CSV文件
    df = pd.read_csv(input_file_path)
    print(f"原始数据行数: {len(df)}")

    # 2. 过滤掉关键字段为空的行（保持与您Cypher语句一致的逻辑）
    df_cleaned = df.dropna(subset=['head', 'tail', 'relation'])
    print(f"过滤掉空值后的行数: {len(df_cleaned)}")

    # 3. 应用清洗函数到 'head' 和 'tail' 列
    # 这将确保所有ID长度不超过50个字符
    df_cleaned['head'] = df_cleaned['head'].apply(clean_and_truncate_id)
    df_cleaned['tail'] = df_cleaned['tail'].apply(clean_and_truncate_id)

    # 4. (可选但推荐) 再次检查并丢弃因截断而可能产生的空字符串（极少数情况）
    # 例如，如果原始id全是空格，截断后会变成空字符串
    before_drop = len(df_cleaned)
    df_cleaned = df_cleaned[(df_cleaned['head'] != '') & (df_cleaned['tail'] != '')]
    after_drop = len(df_cleaned)
    if before_drop != after_drop:
        print(f"警告: 移除了 {before_drop - after_drop} 行因截断产生空ID的记录。")

    # 5. 保存清洗后的数据到新文件
    df_cleaned.to_csv(output_file_path, index=False)
    print(f"数据清洗完成！已保存到: {output_file_path}")
    print(f"最终有效行数: {len(df_cleaned)}")

    # 6. (可选) 打印一些样本以供检查
    print("\n清洗后数据样本预览（前5行）:")
    print(df_cleaned.head())

except FileNotFoundError:
    print(f"错误：找不到输入文件 '{input_file_path}'，请检查文件路径。")
except KeyError as e:
    print(f"错误：CSV文件中找不到必需的列 {e}。请确保您的文件包含 'head', 'tail', 'relation' 列。")
except Exception as e:
    print(f"清洗过程中发生未知错误: {e}")