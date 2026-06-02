"""
Chunk 3: 双路向量化与 ChromaDB 入库
- 对 translated_text 向量化 -> tcm_translated collection（主检索路径）
- 对 original_text 向量化  -> tcm_original collection（辅助/古文直搜路径）
- 两者通过同一个 chunk_id 配对，original_text 存为 metadata 随时可取回
"""
import json
from typing import List, Dict
from loguru import logger
import chromadb
from chromadb.config import Settings
import ollama
from src.config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_ORIGINAL,
    CHROMA_COLLECTION_TRANSLATED,
    EMBED_MODEL,
)


def get_chroma_client() -> chromadb.ClientAPI:
    """获取或创建 ChromaDB 持久化客户端"""
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client


def embed_text(text: str) -> List[float]:
    """调用本地 Ollama nomic-embed-text 生成向量"""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def _build_metadata(chunk: Dict) -> Dict:
    """
    构建存入 ChromaDB 的 Metadata payload
    - 包含古文原文、白话文翻译、书目信息，以便检索命中后直接提取
    - entities 序列化为 JSON 字符串（ChromaDB 不支持嵌套对象）
    """
    return {
        "chunk_id":       chunk["chunk_id"],
        "book":           chunk.get("book", ""),
        "author":         chunk.get("author", ""),
        "chapter":        chunk.get("chapter", ""),
        "article_id":     chunk.get("article_id", ""),
        "original_text":  chunk.get("original_text", ""),
        "translated_text":chunk.get("translated_text", ""),
        "entities_json":  json.dumps(chunk.get("entities", {}), ensure_ascii=False),
        "source_file":    chunk.get("source_file", ""),
    }


def ingest_chunks(chunks: List[Dict], batch_size: int = 50):
    """
    将所有 Chunk 向量化并存入 ChromaDB 双集合
    - tcm_translated: 用白话文向量（主检索）
    - tcm_original:   用古文向量（辅助检索）
    """
    client = get_chroma_client()

    col_translated = client.get_or_create_collection(
        name=CHROMA_COLLECTION_TRANSLATED,
        metadata={"hnsw:space": "cosine"},
    )
    col_original = client.get_or_create_collection(
        name=CHROMA_COLLECTION_ORIGINAL,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(chunks)
    logger.info(f"开始向量化入库，共 {total} 个 Chunk...")

    for i in range(0, total, batch_size):
        batch = chunks[i: i + batch_size]
        logger.info(f"向量化第 {i+1}~{min(i+batch_size, total)}/{total}...")

        ids, trans_embeds, orig_embeds, metadatas, documents = [], [], [], [], []

        for chunk in batch:
            chunk_id = chunk["chunk_id"]
            original = chunk.get("original_text", "")
            translated = chunk.get("translated_text", original)
            meta = _build_metadata(chunk)

            try:
                trans_vec = embed_text(translated)
                orig_vec  = embed_text(original)
            except Exception as e:
                logger.error(f"向量化失败 {chunk_id}: {e}")
                continue

            ids.append(chunk_id)
            trans_embeds.append(trans_vec)
            orig_embeds.append(orig_vec)
            metadatas.append(meta)
            documents.append(translated)  # ChromaDB document 字段存白话文

        if not ids:
            continue

        # 存入白话文向量集合
        col_translated.upsert(
            ids=ids,
            embeddings=trans_embeds,
            documents=documents,
            metadatas=metadatas,
        )

        # 存入古文向量集合（document 字段存古文）
        col_original.upsert(
            ids=ids,
            embeddings=orig_embeds,
            documents=[c.get("original_text", "") for c in batch if c["chunk_id"] in ids],
            metadatas=metadatas,
        )

    logger.info(f"入库完成！tcm_translated: {col_translated.count()} 条，tcm_original: {col_original.count()} 条")


def get_collection_stats() -> Dict:
    """查询当前数据库状态"""
    client = get_chroma_client()
    stats = {}
    for name in [CHROMA_COLLECTION_TRANSLATED, CHROMA_COLLECTION_ORIGINAL]:
        try:
            col = client.get_collection(name)
            stats[name] = col.count()
        except Exception:
            stats[name] = 0
    return stats
