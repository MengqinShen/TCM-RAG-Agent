"""
数据摄取管道入口脚本
运行此脚本以完成：
1. 解析 data/ 目录下所有古籍文档
2. 调用 Ollama 进行白话文翻译 + NER
3. 双路向量化并入库 ChromaDB

用法:
    python run_ingestion.py               # 全量处理
    python run_ingestion.py --limit 50    # 仅处理前50条（用于快速测试）
    python run_ingestion.py --skip-translate  # 跳过翻译（仅用原文向量）
"""
import argparse
import json
from pathlib import Path
from loguru import logger

from src.ingestion.chunker import parse_all_documents, save_chunks_to_jsonl
from src.ingestion.translator import translate_and_annotate_all
from src.ingestion.embedder import ingest_chunks, get_collection_stats
from src.config import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="TCM 知识库摄取管道")
    parser.add_argument("--limit", type=int, default=None, help="限制处理条数（测试用）")
    parser.add_argument("--skip-translate", action="store_true", help="跳过 LLM 翻译步骤")
    parser.add_argument("--from-cache", type=str, default=None, help="从已有 JSONL 缓存文件恢复")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🏥 中医古籍智能问答系统 - 知识库摄取管道启动")
    logger.info("=" * 60)

    # ── Step 1: 解析文档 ──────────────────────────────────────
    if args.from_cache:
        logger.info(f"从缓存加载: {args.from_cache}")
        with open(args.from_cache, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        logger.info(f"缓存加载完成，共 {len(chunks)} 个 Chunk")
    else:
        logger.info(f"Step 1/3: 解析文档目录: {DATA_DIR}")
        chunks = parse_all_documents(DATA_DIR)

        if not chunks:
            logger.error("未解析到任何内容，请检查 data/ 目录是否有支持的文件")
            return

        # 保存解析结果到缓存
        cache_path = "chunks_raw.jsonl"
        save_chunks_to_jsonl(chunks, cache_path)
        logger.info(f"原始 Chunks 已缓存至: {cache_path}")

    # 限制处理数量
    if args.limit:
        chunks = chunks[:args.limit]
        logger.info(f"⚠️  限制模式：仅处理前 {args.limit} 条")

    # ── Step 2: 翻译 + NER ───────────────────────────────────
    if args.skip_translate:
        logger.info("Step 2/3: 跳过翻译，使用原文作为白话文字段")
        for chunk in chunks:
            if "translated_text" not in chunk:
                chunk["translated_text"] = chunk["original_text"]
            if "entities" not in chunk:
                chunk["entities"] = {
                    "symptoms": [], "diseases": [], "herbs": [],
                    "acupoints": [], "treatments": [], "formulas": []
                }
    else:
        logger.info(f"Step 2/3: 开始翻译与实体识别（共 {len(chunks)} 条，请耐心等待...）")
        chunks = translate_and_annotate_all(chunks, batch_size=5, delay=0.3)

        # 保存翻译结果到缓存
        translated_cache = "chunks_translated.jsonl"
        save_chunks_to_jsonl(chunks, translated_cache)
        logger.info(f"翻译结果已缓存至: {translated_cache}")

    # ── Step 3: 向量化入库 ───────────────────────────────────
    logger.info(f"Step 3/3: 向量化并写入 ChromaDB...")
    ingest_chunks(chunks, batch_size=50)

    # ── 完成统计 ─────────────────────────────────────────────
    stats = get_collection_stats()
    logger.info("=" * 60)
    logger.info("✅ 摄取完成！数据库状态：")
    for name, count in stats.items():
        logger.info(f"   {name}: {count} 条向量")
    logger.info("=" * 60)
    logger.info("📌 下一步：运行 Streamlit 前端")
    logger.info("   streamlit run src/frontend/app.py")


if __name__ == "__main__":
    main()
