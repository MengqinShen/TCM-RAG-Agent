"""
Chunk 2: 古文白话文翻译管道
调用本地 Ollama (qwen3.5) 对每条古文条文进行批量翻译与注释
"""
import json
import time
from typing import List, Dict
from loguru import logger
import ollama
from src.config import LLM_MODEL


TRANSLATE_SYSTEM_PROMPT = """你是一位精通中医古籍的专家学者。
你的任务是将输入的中医古文条文翻译成现代临床医学语言的白话文。

翻译要求：
1. 保留所有中医专业术语（如"太阳病"、"少阴证"等）不翻译，直接保留原词
2. 将症状描述转换为现代语言（如"恶寒"->怕冷，"头项强痛"->头颈部僵硬疼痛）
3. 只输出白话文翻译，不要加额外解释或评论
4. 输出长度控制在原文 2-3 倍以内"""

NER_SYSTEM_PROMPT = """你是一位中医命名实体识别专家。
从给定的中医古文中提取以下类型的实体，以JSON格式输出：
- symptoms: 症状（如发热、恶寒）
- diseases: 病名/证型（如太阳病、消渴）
- herbs: 药材（如麻黄、桂枝）
- acupoints: 穴位（如足三里、合谷）
- treatments: 治法（如发汗、清热）
- formulas: 方剂（如麻黄汤、桂枝汤）

只输出合法的JSON，不要有任何额外文字。示例格式：
{"symptoms": ["发热", "恶寒"], "diseases": ["太阳病"], "herbs": ["麻黄"], "acupoints": [], "treatments": ["发汗"], "formulas": ["麻黄汤"]}"""


def translate_chunk(chunk: Dict) -> Dict:
    """对单条 Chunk 进行白话文翻译"""
    original = chunk["original_text"]

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": f"请翻译以下古文：\n\n{original}"},
            ],
            options={"temperature": 0.1},  # 低温度确保稳定翻译
        )
        translated = response["message"]["content"].strip()
        chunk["translated_text"] = translated
        logger.debug(f"[翻译] {chunk['chunk_id']}: OK")
    except Exception as e:
        logger.error(f"[翻译失败] {chunk['chunk_id']}: {e}")
        chunk["translated_text"] = original  # fallback: 保留原文

    return chunk


def extract_entities(chunk: Dict) -> Dict:
    """对单条 Chunk 进行命名实体识别"""
    original = chunk["original_text"]

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": NER_SYSTEM_PROMPT},
                {"role": "user", "content": original},
            ],
            options={"temperature": 0.0},
        )
        raw = response["message"]["content"].strip()
        # 清理可能的 markdown 代码块
        raw = raw.replace("```json", "").replace("```", "").strip()
        entities = json.loads(raw)
        chunk["entities"] = entities
        logger.debug(f"[NER] {chunk['chunk_id']}: {entities}")
    except Exception as e:
        logger.warning(f"[NER失败] {chunk['chunk_id']}: {e}")
        chunk["entities"] = {
            "symptoms": [], "diseases": [], "herbs": [],
            "acupoints": [], "treatments": [], "formulas": []
        }

    return chunk


def translate_and_annotate_all(
    chunks: List[Dict],
    batch_size: int = 5,
    delay: float = 0.5,
) -> List[Dict]:
    """
    批量翻译 + NER 全部 Chunks
    batch_size: 每批处理数量（控制内存与速率）
    delay: 批次间延迟（秒），避免过载本地 Ollama
    """
    total = len(chunks)
    results = []

    logger.info(f"开始翻译与实体识别，共 {total} 个 Chunk，批大小={batch_size}")

    for i in range(0, total, batch_size):
        batch = chunks[i: i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"处理第 {batch_num} 批 ({i+1}~{min(i+batch_size, total)}/{total})...")

        for chunk in batch:
            chunk = translate_chunk(chunk)
            chunk = extract_entities(chunk)
            results.append(chunk)

        if i + batch_size < total:
            time.sleep(delay)

    logger.info(f"翻译与实体识别完成，共处理 {len(results)} 个 Chunk")
    return results
