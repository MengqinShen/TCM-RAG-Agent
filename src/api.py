"""
FastAPI 后端 API
提供 /query 和 /ingest/status 端点
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger

from src.retrieval.query_processor import process_query
from src.retrieval.retriever import hybrid_retrieve
from src.generation.generator import generate_answer
from src.ingestion.embedder import get_collection_stats

app = FastAPI(
    title="中医古籍智能问答系统 API",
    description="基于混合 RAG + LangChain + Ollama 的中医辅助诊疗接口",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求 / 响应模型 ────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class CitationItem(BaseModel):
    chunk_id: str
    book: str
    author: str
    chapter: str
    article_id: str
    original_text: str
    translated_text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    is_confident: bool
    confidence: float
    citations: List[str]
    evidence: List[CitationItem]
    intent: str
    expanded_query: str


# ── 端点 ───────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "TCM RAG API v0.1"}


@app.get("/health")
def health():
    stats = get_collection_stats()
    return {
        "status": "ok",
        "db_stats": stats,
    }


@app.post("/api/v1/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    主查询接口：接受白话文提问，返回古籍依据的诊疗建议
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    logger.info(f"[API] 收到查询: {req.query[:80]}")

    # Step 1: 查询处理 + 扩展
    processed = process_query(req.query)

    # Step 2: 混合检索
    retrieval = hybrid_retrieve(processed)

    # Step 3: 生成回答
    result = generate_answer(req.query, retrieval)

    # 构建 evidence 列表
    evidence = [
        CitationItem(
            chunk_id=c["chunk_id"],
            book=c["book"],
            author=c["author"],
            chapter=c["chapter"],
            article_id=c["article_id"],
            original_text=c["original_text"],
            translated_text=c["translated_text"],
            score=c.get("final_score", c.get("score", 0.0)),
        )
        for c in result["chunks"]
    ]

    return QueryResponse(
        answer=result["answer"],
        is_confident=result["is_confident"],
        confidence=retrieval["confidence"],
        citations=result["citations"],
        evidence=evidence,
        intent=processed["intent"],
        expanded_query=processed["expanded_query"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
