"""
Retrieval Layer 1: 查询处理器
- 对用户白话文提问进行意图分析
- 通过古今术语对照词典扩展查询词（Query Expansion）
"""
import json
import re
from typing import List, Dict, Tuple
from pathlib import Path
from loguru import logger
import ollama
from src.config import LLM_MODEL, ONTOLOGY_PATH


# 加载古今术语对照词典
def _load_ontology() -> Dict:
    path = Path(ONTOLOGY_PATH)
    if not path.exists():
        logger.warning(f"词典文件不存在: {ONTOLOGY_PATH}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


ONTOLOGY = _load_ontology()

NER_QUERY_PROMPT = """你是一位中医专家，请从用户输入的症状描述中提取关键医学信息。
以JSON格式输出，包含以下字段：
- intent: 用户意图（"诊断" / "治疗" / "方剂查询" / "文献溯源"）
- symptoms: 症状列表（现代白话文）
- diseases: 涉及的疾病名称（若有）
- herbs: 涉及的草药名称（若有）

只输出JSON，不加任何解释。示例：
{"intent": "诊断", "symptoms": ["发烧", "怕冷", "头痛"], "diseases": [], "herbs": []}"""


def analyze_query(query: str) -> Dict:
    """调用 LLM 分析用户查询意图和实体"""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": NER_QUERY_PROMPT},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.0},
        )
        raw = response["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        logger.debug(f"[查询分析] {result}")
        return result
    except Exception as e:
        logger.warning(f"[查询分析失败] {e}，使用默认值")
        return {"intent": "诊断", "symptoms": [], "diseases": [], "herbs": []}


def expand_query(query_analysis: Dict) -> Tuple[str, List[str]]:
    """
    将现代医学术语扩展为古籍术语
    返回: (扩展后的检索字符串, 扩展出的古籍关键词列表)
    """
    modern_terms = (
        query_analysis.get("symptoms", []) +
        query_analysis.get("diseases", []) +
        query_analysis.get("herbs", [])
    )

    classical_terms = []

    # 遍历所有分类的映射
    all_mappings = {}
    for category in ["symptoms", "diseases", "herbs", "treatments", "syndromes"]:
        all_mappings.update(ONTOLOGY.get(category, {}))

    for term in modern_terms:
        # 精确匹配
        if term in all_mappings:
            classical_terms.extend(all_mappings[term])
        else:
            # 模糊匹配：检查是否有包含关系
            for key, values in all_mappings.items():
                if key in term or term in key:
                    classical_terms.extend(values)

    # 去重
    classical_terms = list(set(classical_terms))

    # 构建扩展查询字符串（现代词 + 古籍词）
    all_terms = list(set(modern_terms + classical_terms))
    expanded_query = " ".join(all_terms)

    logger.info(f"[查询扩展] 现代词: {modern_terms} -> 古籍词: {classical_terms}")
    return expanded_query, classical_terms


def process_query(user_query: str) -> Dict:
    """
    完整查询处理流程入口
    输入: 用户原始白话文问题
    输出: {
        original_query, intent, symptoms, diseases, herbs,
        expanded_query, classical_terms
    }
    """
    analysis = analyze_query(user_query)
    expanded_query, classical_terms = expand_query(analysis)

    return {
        "original_query":  user_query,
        "intent":          analysis.get("intent", "诊断"),
        "symptoms":        analysis.get("symptoms", []),
        "diseases":        analysis.get("diseases", []),
        "herbs":           analysis.get("herbs", []),
        "expanded_query":  expanded_query,
        "classical_terms": classical_terms,
    }
