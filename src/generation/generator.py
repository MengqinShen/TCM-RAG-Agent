"""
Generation Layer: 回答生成器
- 构建含古籍证据的 CoT 提示词
- 调用 Ollama LLM 生成规范化回答
- 低置信度时触发拒答机制
"""
from typing import List, Dict
from loguru import logger
import ollama
from src.config import LLM_MODEL, CONFIDENCE_THRESHOLD


SYSTEM_PROMPT = """你是一位严谨的中医临床辅助专家，精通中医古籍。

你必须遵守以下规则：
1. **只基于提供的[参考古籍]内容作答**，绝对不允许凭空捏造。
2. 使用现代临床医学语言解释症候和治法，便于医生理解。
3. 每一条诊断分析、方剂推荐，必须强制附上古籍引用，格式为：
   **【来源：《书名》·篇章 第N条，作者】**
4. 如参考资料不足以支撑判断，明确告知"文献依据不足"，不要强行作答。
5. 回答结构：
   - **证型判断**（依据症状推断最可能的中医证型）
   - **治法建议**（根据证型给出治则）
   - **推荐方剂**（若古籍有明确记载）
   - **古籍原文引用**（每条有据可查的证据）

【重要】：你是辅助工具，最终诊断由医生负责。"""


def _format_evidence(chunks: List[Dict]) -> str:
    """将检索到的 Chunks 格式化为 LLM 上下文证据块"""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        citation = f"《{chunk['book']}》·{chunk['chapter']}（{chunk['author']}）第{chunk['article_id']}条"
        parts.append(
            f"[参考{i}] 来源：{citation}\n"
            f"  古文原文：{chunk['original_text']}\n"
            f"  白话解释：{chunk['translated_text']}"
        )
    return "\n\n".join(parts)


def generate_answer(user_query: str, retrieval_result: Dict) -> Dict:
    """
    生成最终回答
    输入: 用户查询 + hybrid_retrieve() 的输出
    输出: {
        "answer": str,          # LLM 生成的回答
        "citations": List[str], # 引用列表
        "is_confident": bool,   # 是否基于充分证据
        "chunks": List[Dict],   # 支撑证据的原始 chunks
    }
    """
    is_confident = retrieval_result.get("is_confident", False)
    chunks = retrieval_result.get("results", [])
    confidence = retrieval_result.get("confidence", 0.0)

    # ── 置信度不足，直接拒答 ──────────────────────────────────
    if not is_confident or not chunks:
        logger.warning(f"[拒答] 置信度 {confidence:.3f} < 阈值 {CONFIDENCE_THRESHOLD}")
        return {
            "answer": (
                "⚠️ **【文献中未找到高置信度相关记载】**\n\n"
                f"当前查询在已接入的古籍知识库中匹配度较低（相似度：{confidence:.2f}），"
                "为确保医疗安全，系统停止生成诊断建议。\n\n"
                "**建议：**\n"
                "- 尝试使用更具体的症状描述（如舌象、脉象）\n"
                "- 补充更多病征信息\n"
                "- 直接查阅相关古籍原文"
            ),
            "citations": [],
            "is_confident": False,
            "chunks": [],
        }

    # ── 构建证据上下文 ────────────────────────────────────────
    evidence_text = _format_evidence(chunks)
    citations = [
        f"《{c['book']}》·{c['chapter']}（{c['author']}）第{c['article_id']}条"
        for c in chunks
    ]

    user_message = (
        f"【参考古籍原文及白话解释】\n{evidence_text}\n\n"
        f"【医生提问】\n{user_query}"
    )

    # ── 调用 LLM 生成 ─────────────────────────────────────────
    logger.info(f"[生成] 调用 {LLM_MODEL}，证据 {len(chunks)} 条...")
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            options={
                "temperature": 0.2,   # 略高于0，保留专业表达的流畅度
                "top_p": 0.9,
            },
        )
        answer = response["message"]["content"].strip()
        logger.info(f"[生成] 完成，回答长度: {len(answer)} 字")
    except Exception as e:
        logger.error(f"[生成失败] {e}")
        answer = f"❌ 生成服务异常，请检查 Ollama 是否正常运行。错误：{e}"

    return {
        "answer":       answer,
        "citations":    citations,
        "is_confident": True,
        "chunks":       chunks,
    }
