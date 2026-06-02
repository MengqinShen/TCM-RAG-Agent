"""
Retrieval Layer 2: 混合检索器
- Path A: ChromaDB 向量相似度检索（语义）
- Path B: BM25 关键词检索（精确词匹配，针对中医专有名词）
- 合并去重后返回 Top-K 候选段落
"""
import json
from typing import List, Dict, Optional
from loguru import logger
import chromadb
from chromadb.config import Settings
import ollama

from src.config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_TRANSLATED,
    CHROMA_COLLECTION_ORIGINAL,
    EMBED_MODEL,
    TOP_K_VECTOR,
    TOP_K_FINAL,
    CONFIDENCE_THRESHOLD,
)


# ── ChromaDB Client（单例） ─────────────────────────────────────
_chroma_client: Optional[chromadb.ClientAPI] = None


def _get_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def _embed(text: str) -> List[float]:
    """生成查询向量"""
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]


# ── Path A: 向量检索 ───────────────────────────────────────────

def vector_search(query: str, collection_name: str, top_k: int = TOP_K_VECTOR) -> List[Dict]:
    """
    在指定 ChromaDB 集合中进行余弦相似度检索
    返回标准化的 chunk 列表（含 distance 作为相似度分数）
    """
    client = _get_client()
    try:
        col = client.get_collection(collection_name)
    except Exception:
        logger.warning(f"集合 {collection_name} 不存在，跳过向量检索")
        return []

    query_vec = _embed(query)
    results = col.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, col.count()),
        include=["metadatas", "documents", "distances"],
    )

    chunks = []
    for i, meta in enumerate(results["metadatas"][0]):
        distance = results["distances"][0][i]
        # ChromaDB cosine distance: 0=完全相同, 2=完全相反
        # 转换为相似度分数 [0,1]
        score = 1.0 - (distance / 2.0)
        chunk = {
            "chunk_id":       meta.get("chunk_id", f"vec_{i}"),
            "book":           meta.get("book", ""),
            "author":         meta.get("author", ""),
            "chapter":        meta.get("chapter", ""),
            "article_id":     meta.get("article_id", ""),
            "original_text":  meta.get("original_text", ""),
            "translated_text":meta.get("translated_text", ""),
            "entities":       json.loads(meta.get("entities_json", "{}")),
            "score":          round(score, 4),
            "source":         f"vector:{collection_name}",
        }
        chunks.append(chunk)

    return chunks


# ── Path B: BM25 关键词检索（基于 ChromaDB where 过滤） ─────────

def keyword_search(classical_terms: List[str], top_k: int = TOP_K_VECTOR) -> List[Dict]:
    """
    使用古籍关键词对 ChromaDB 原文集合进行关键词过滤检索
    利用 ChromaDB 的 where_document 参数做文本包含匹配
    模拟 BM25 的精确词匹配效果（V0 简化实现）
    """
    if not classical_terms:
        return []

    client = _get_client()
    try:
        col = client.get_collection(CHROMA_COLLECTION_ORIGINAL)
    except Exception:
        logger.warning("tcm_original 集合不存在，跳过关键词检索")
        return []

    all_results = []
    seen_ids = set()

    for term in classical_terms[:5]:  # 最多用 5 个关键词
        try:
            results = col.get(
                where_document={"$contains": term},
                limit=top_k,
                include=["metadatas", "documents"],
            )
            for i, meta in enumerate(results.get("metadatas", [])):
                cid = meta.get("chunk_id", f"kw_{i}")
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                all_results.append({
                    "chunk_id":       meta.get("chunk_id", ""),
                    "book":           meta.get("book", ""),
                    "author":         meta.get("author", ""),
                    "chapter":        meta.get("chapter", ""),
                    "article_id":     meta.get("article_id", ""),
                    "original_text":  meta.get("original_text", ""),
                    "translated_text":meta.get("translated_text", ""),
                    "entities":       json.loads(meta.get("entities_json", "{}")),
                    "score":          0.6,   # 关键词命中给予固定基础分
                    "source":         f"keyword:{term}",
                })
        except Exception as e:
            logger.warning(f"关键词检索失败 '{term}': {e}")

    return all_results


# ── 合并 + 重排 ────────────────────────────────────────────────

def _deduplicate_and_merge(vec_results: List[Dict], kw_results: List[Dict]) -> List[Dict]:
    """合并两路结果，关键词命中可提升分数"""
    merged: Dict[str, Dict] = {}

    for chunk in vec_results:
        merged[chunk["chunk_id"]] = chunk

    for chunk in kw_results:
        cid = chunk["chunk_id"]
        if cid in merged:
            # 同时被向量和关键词命中，分数提升
            merged[cid]["score"] = min(merged[cid]["score"] + 0.15, 1.0)
            merged[cid]["source"] += f" + {chunk['source']}"
        else:
            merged[cid] = chunk

    return list(merged.values())


def _simple_rerank(chunks: List[Dict], query: str) -> List[Dict]:
    """
    简单重排：综合向量分数 + 是否包含查询关键词
    V0 阶段使用规则重排，后续可替换为 Cross-Encoder 模型
    """
    query_terms = set(query.split())
    for chunk in chunks:
        bonus = 0.0
        text = chunk["original_text"] + chunk["translated_text"]
        for term in query_terms:
            if len(term) > 1 and term in text:
                bonus += 0.02
        chunk["final_score"] = round(min(chunk["score"] + bonus, 1.0), 4)

    return sorted(chunks, key=lambda x: x["final_score"], reverse=True)


# ── 主检索入口 ─────────────────────────────────────────────────

def hybrid_retrieve(processed_query: Dict) -> Dict:
    """
    混合检索主入口
    输入: process_query() 的输出
    输出: {
        "results": List[chunk],    # Top-K 最终候选
        "confidence": float,       # 最高分（用于拒答判断）
        "is_confident": bool,      # 是否超过置信度阈值
        "expanded_query": str,
    }
    """
    expanded_query = processed_query.get("expanded_query", processed_query["original_query"])
    classical_terms = processed_query.get("classical_terms", [])

    logger.info(f"[检索] 扩展查询词: {expanded_query[:80]}...")

    # Path A: 白话文向量检索（主路径）
    vec_results = vector_search(expanded_query, CHROMA_COLLECTION_TRANSLATED, top_k=TOP_K_VECTOR)
    logger.info(f"[向量检索] 召回 {len(vec_results)} 条")

    # Path B: 古籍关键词精确检索
    kw_results = keyword_search(classical_terms, top_k=TOP_K_VECTOR)
    logger.info(f"[关键词检索] 召回 {len(kw_results)} 条")

    # 合并去重
    merged = _deduplicate_and_merge(vec_results, kw_results)

    # 简单重排
    ranked = _simple_rerank(merged, expanded_query)

    # 取 Top-K
    top_results = ranked[:TOP_K_FINAL]

    # 置信度判断
    confidence = top_results[0]["final_score"] if top_results else 0.0
    is_confident = confidence >= CONFIDENCE_THRESHOLD

    logger.info(
        f"[检索完成] 最终候选 {len(top_results)} 条，"
        f"最高分={confidence:.3f}，置信={'✅' if is_confident else '❌'}"
    )

    return {
        "results":        top_results,
        "confidence":     confidence,
        "is_confident":   is_confident,
        "expanded_query": expanded_query,
    }
