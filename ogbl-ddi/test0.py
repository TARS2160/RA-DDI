import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
from typing import List, Dict, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DDIRelationExtractor:
    def __init__(self):
        """初始化模型和组件"""
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"使用设备: {'GPU' if self.device == 0 else 'CPU'}")

        # 加载生物医学NER模型
        try:
            self.ner_pipeline = pipeline(
                "token-classification",
                model="D://study/gnn/DDKG-main/DDKG-main/models/huggingface/dmis-lab_biobert-v1.1",
                aggregation_strategy="simple",
                device=self.device
            )
        except:
            # 备用模型
            self.ner_pipeline = pipeline(
                "token-classification",
                model="D://study/gnn/DDKG-main/DDKG-main/models/huggingface/bert-base-uncased",
                aggregation_strategy="simple",
                device=self.device
            )

        # 关系关键词词典
        self.relationship_patterns = {
            'increase': ['increase', 'enhance', 'potentiate', 'augment', 'boost', 'elevate', 'intensify'],
            'decrease': ['decrease', 'reduce', 'inhibit', 'diminish', 'suppress', 'lower', 'block'],
            'interaction': ['interact', 'affect', 'alter', 'change', 'modify', 'influence'],
            'adverse': ['adverse', 'side effect', 'toxicity', 'risk', 'danger'],
            'synergistic': ['synerg', 'cooperat', 'combined effect']
        }

    def extract_entities(self, text: str) -> List[Dict]:
        """提取文本中的实体"""
        try:
            entities = self.ner_pipeline(text)
            return entities
        except Exception as e:
            logger.warning(f"实体提取失败: {e}")
            return []

    def analyze_relationship(self, description: str, drug1: str, drug2: str) -> str:
        """分析药物间关系"""
        description_lower = description.lower()
        drug1_lower = drug1.lower()
        drug2_lower = drug2.lower()

        # 模式1: Drug2 [action] Drug1's effects
        pattern1 = rf"{re.escape(drug2_lower)}.*?({'|'.join(self.relationship_patterns['increase'])})"
        pattern2 = rf"{re.escape(drug2_lower)}.*?({'|'.join(self.relationship_patterns['decrease'])})"

        # 模式2: [action] of Drug1 by Drug2
        pattern3 = rf"({'|'.join(self.relationship_patterns['increase'])}).*?{re.escape(drug1_lower)}.*?{re.escape(drug2_lower)}"
        pattern4 = rf"({'|'.join(self.relationship_patterns['decrease'])}).*?{re.escape(drug1_lower)}.*?{re.escape(drug2_lower)}"

        # 检查增加关系
        if (re.search(pattern1, description_lower) or
                re.search(pattern3, description_lower)):
            return "increase"

        # 检查减少关系
        if (re.search(pattern2, description_lower) or
                re.search(pattern4, description_lower)):
            return "decrease"

        # 检查其他类型的关系
        for rel_type, keywords in self.relationship_patterns.items():
            if rel_type in ['increase', 'decrease']:
                continue
            if any(keyword in description_lower for keyword in keywords):
                return rel_type

        return "unknown"

    def extract_relationship_strength(self, description: str) -> str:
        """提取关系强度（可选）"""
        strength_indicators = {
            'strong': ['significantly', 'markedly', 'substantially', 'greatly'],
            'moderate': ['moderately', 'modest', 'noticeably'],
            'weak': ['slightly', 'mildly', 'minimally']
        }

        description_lower = description.lower()
        for strength, indicators in strength_indicators.items():
            if any(indicator in description_lower for indicator in indicators):
                return strength
        return "unspecified"

    def process_description(self, description: str, drug1: str, drug2: str) -> Dict:
        """处理单个描述文本"""
        # 基本关系分析
        relationship = self.analyze_relationship(description, drug1, drug2)

        # 提取实体（用于验证）
        entities = self.extract_entities(description)

        # 提取关系强度
        strength = self.extract_relationship_strength(description)

        return {
            'relationship': relationship,
            'strength': strength,
            'entities_found': len(entities),
            'confidence': self.calculate_confidence(description, relationship)
        }

    def calculate_confidence(self, description: str, relationship: str) -> float:
        """计算预测置信度"""
        if relationship == "unknown":
            return 0.3

        description_lower = description.lower()
        confidence = 0.7  # 基础置信度

        # 基于关键词明确性调整置信度
        strong_indicators = ['may', 'can', 'will', 'should']
        weak_indicators = ['might', 'could', 'possibly', 'potentially']

        if any(indicator in description_lower for indicator in strong_indicators):
            confidence += 0.2
        elif any(indicator in description_lower for indicator in weak_indicators):
            confidence -= 0.1

        return min(confidence, 1.0)


def main():
    """主函数"""
    # 初始化提取器
    extractor = DDIRelationExtractor()

    try:
        # 读取输入文件
        logger.info("读取输入文件...")
        df = pd.read_csv('ddi_description.csv')
        logger.info(f"成功读取 {len(df)} 条记录")

    except FileNotFoundError:
        logger.error("文件 'ddi_description.csv' 未找到")
        return
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return

    results = []

    for idx, row in df.iterrows():
        try:
            logger.info(f"处理第 {idx + 1}/{len(df)} 条记录...")

            # 提取关系信息
            result = extractor.process_description(
                row['description'],
                row['first drug name'],
                row['second drug name']
            )

            # 构建输出行
            output_row = {
                'first drug id': row['first drug id'],
                'second drug id': row['second drug id'],
                'label': result['relationship'],
                'strength': result['strength'],
                'confidence': result['confidence'],
                'entities_detected': result['entities_found']
            }

            results.append(output_row)

            # 打印处理进度
            if (idx + 1) % 10 == 0:
                logger.info(f"已处理 {idx + 1} 条记录")

        except Exception as e:
            logger.error(f"处理第 {idx + 1} 条记录时出错: {e}")
            # 添加错误记录
            error_row = {
                'first drug id': row['first drug id'],
                'second drug id': row['second drug id'],
                'label': 'error',
                'strength': 'error',
                'confidence': 0.0,
                'entities_detected': 0
            }
            results.append(error_row)

    # 创建输出DataFrame
    output_df = pd.DataFrame(results)

    # 保存结果
    try:
        output_df.to_csv('test0.csv', index=False)
        logger.info(f"结果已保存到 'test0.csv'")

        # 统计结果
        label_counts = output_df['label'].value_counts()
        logger.info("关系类型统计:")
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count} 条")

    except Exception as e:
        logger.error(f"保存结果失败: {e}")


if __name__ == "__main__":
    main()