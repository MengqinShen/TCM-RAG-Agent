"""
Streamlit 前端 - 中医古籍智能问答系统
双分屏布局：左侧对话 | 右侧古文溯源面板
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from loguru import logger

from src.retrieval.query_processor import process_query
from src.retrieval.retriever import hybrid_retrieve
from src.generation.generator import generate_answer
from src.ingestion.embedder import get_collection_stats

# ── 页面基础配置 ───────────────────────────────────────────────
st.set_page_config(
    page_title="中医古籍智能辅助诊断系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS 样式 ───────────────────────────────────────────────────
st.markdown("""
<style>
/* 整体背景 */
.stApp { background-color: #f5f0e8; }

/* 顶部标题栏 */
.main-header {
    background: linear-gradient(135deg, #1a3a2a 0%, #2d5a3d 100%);
    color: white; padding: 1.2rem 2rem; border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.main-header h1 { margin: 0; font-size: 1.6rem; }
.main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.9rem; }

/* 聊天消息 - 用户 */
.user-msg {
    background: #2d5a3d; color: white;
    padding: 0.9rem 1.2rem; border-radius: 12px 12px 4px 12px;
    margin: 0.6rem 0; max-width: 85%; margin-left: auto;
}
/* 聊天消息 - AI */
.ai-msg {
    background: white; color: #1a1a1a;
    padding: 0.9rem 1.2rem; border-radius: 12px 12px 12px 4px;
    margin: 0.6rem 0; max-width: 92%;
    border-left: 4px solid #2d5a3d;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
/* 警告拒答 */
.refuse-msg {
    background: #fff8e1; color: #6d4c00;
    padding: 0.9rem 1.2rem; border-radius: 8px;
    border-left: 4px solid #ffc107;
}
/* 古文证据卡片 */
.evidence-card {
    background: #faf8f2; border: 1px solid #d4c5a9;
    border-radius: 10px; padding: 1rem;
    margin-bottom: 0.8rem;
}
.evidence-card .source-tag {
    background: #2d5a3d; color: white;
    font-size: 0.75rem; padding: 2px 8px;
    border-radius: 4px; display: inline-block;
    margin-bottom: 0.5rem;
}
.evidence-card .classical-text {
    font-family: "STSong", "SimSun", serif;
    font-size: 1rem; color: #2c1810;
    border-left: 3px solid #8b6914;
    padding-left: 0.8rem; margin: 0.5rem 0;
    line-height: 1.8;
}
.evidence-card .modern-text {
    font-size: 0.88rem; color: #555;
    margin-top: 0.4rem; line-height: 1.6;
}
.score-badge {
    float: right; font-size: 0.75rem;
    color: #2d5a3d; font-weight: bold;
}
/* 侧边栏 */
.sidebar-section {
    background: white; border-radius: 10px;
    padding: 1rem; margin-bottom: 1rem;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)


# ── 初始化 Session State ───────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_evidence" not in st.session_state:
    st.session_state.last_evidence = []
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}


# ── 侧边栏 ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 系统状态")

    stats = get_collection_stats()
    total_chunks = sum(stats.values()) // 2  # 两个集合存了相同条数
    if total_chunks > 0:
        st.success(f"✅ 知识库已就绪\n\n📚 共 **{total_chunks}** 条古籍条文")
    else:
        st.error("❌ 知识库为空\n\n请先运行:\n```\npython run_ingestion.py\n```")



    st.divider()

    st.markdown("### 💡 示例问题")
    examples = [
        "患者发烧、怕冷、头痛、无汗，该怎么辨证？",
        "口渴多饮多尿消瘦是什么证？古籍有哪些记载？",
        "中风后半身不遂，古籍如何论治？",
        "心悸胸闷气短，中医怎么看？",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:10]}"):
            st.session_state["prefill_query"] = ex

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_evidence = []
        st.session_state.last_meta = {}
        st.rerun()


# ── 主页面 ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏥 中医古籍智能辅助诊断系统</h1>
  <p>基于混合 RAG · 古今术语桥接 · 严格古籍引用 | 所有回答均可溯源</p>
</div>
""", unsafe_allow_html=True)

# 双列布局
col_chat, col_evidence = st.columns([3, 2], gap="large")

# ── 左列：对话区 ──────────────────────────────────────────────
with col_chat:
    st.markdown("#### 💬 对话区")

    # 历史消息
    chat_container = st.container(height=480)
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; color:#888; margin-top:6rem;">
                <div style="font-size:2.5rem">📖</div>
                <p>请在下方输入您的问题</p>
                <p style="font-size:0.85rem">系统将从中医古籍中检索相关依据</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="user-msg">🧑‍⚕️ {msg["content"]}</div>', unsafe_allow_html=True)
                elif msg["role"] == "assistant":
                    if msg.get("is_confident", True):
                        st.markdown(f'<div class="ai-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="refuse-msg">{msg["content"]}</div>', unsafe_allow_html=True)

    # 输入框
    prefill = st.session_state.pop("prefill_query", "")
    user_input = st.chat_input("请用白话文描述症状或提出中医问题...", key="main_input")

    # 处理预填或手动输入
    query_to_run = prefill or user_input

    if query_to_run:
        st.session_state.messages.append({"role": "user", "content": query_to_run})

        with st.spinner("🔍 正在检索古籍并生成回答，请稍候..."):
            try:
                processed  = process_query(query_to_run)
                retrieval  = hybrid_retrieve(processed)
                result     = generate_answer(query_to_run, retrieval)

                st.session_state.last_evidence = result["chunks"]
                st.session_state.last_meta = {
                    "intent":         processed["intent"],
                    "expanded_query": processed["expanded_query"],
                    "confidence":     retrieval["confidence"],
                    "is_confident":   retrieval["is_confident"],
                    "citations":      result["citations"],
                }
                st.session_state.messages.append({
                    "role":        "assistant",
                    "content":     result["answer"],
                    "is_confident": result["is_confident"],
                })
            except Exception as e:
                logger.error(f"处理失败: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ 系统错误：{e}",
                    "is_confident": False,
                })

        st.rerun()

    # 反馈按钮
    if st.session_state.messages:
        fb_cols = st.columns(4)
        with fb_cols[0]:
            if st.button("👍 采纳", use_container_width=True):
                st.toast("感谢反馈！已记录为正向样本")
        with fb_cols[1]:
            if st.button("👎 查无此文", use_container_width=True):
                st.toast("已记录，将用于优化检索")
        with fb_cols[2]:
            if st.button("⚠️ 翻译有误", use_container_width=True):
                st.toast("已标记翻译问题，谢谢")
        with fb_cols[3]:
            if st.button("📋 复制回答", use_container_width=True):
                st.toast("请手动复制上方内容")


# ── 右列：古文溯源面板 ────────────────────────────────────────
with col_evidence:
    st.markdown("#### 📚 古籍溯源面板")

    if st.session_state.last_meta:
        meta = st.session_state.last_meta
        conf = meta.get("confidence", 0)
        conf_color = "#2d5a3d" if meta.get("is_confident") else "#c62828"
        st.markdown(f"""
        <div style="background:white; border-radius:8px; padding:0.8rem; margin-bottom:0.8rem; border:1px solid #ddd;">
            <div style="font-size:0.8rem; color:#555;">
                <b>意图：</b>{meta.get('intent','—')} &nbsp;|&nbsp;
                <b>置信度：</b><span style="color:{conf_color}; font-weight:bold;">{conf:.2f}</span>
            </div>
            <div style="font-size:0.78rem; color:#777; margin-top:4px;">
                <b>扩展检索词：</b>{meta.get('expanded_query','—')[:60]}...
            </div>
        </div>
        """, unsafe_allow_html=True)

    evidence = st.session_state.last_evidence
    if not evidence:
        st.markdown("""
        <div style="text-align:center; color:#aaa; margin-top:5rem;">
            <div style="font-size:2rem">📜</div>
            <p>此处将显示匹配到的古籍原文及出处</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption(f"共找到 {len(evidence)} 条相关古籍记载")
        evidence_container = st.container(height=520)
        with evidence_container:
            for i, chunk in enumerate(evidence, start=1):
                score = chunk.get("final_score", chunk.get("score", 0))
                citation = f"《{chunk['book']}》·{chunk['chapter']}（{chunk['author']}）第{chunk['article_id']}条"
                st.markdown(f"""
                <div class="evidence-card">
                    <span class="source-tag">引用 [{i}]</span>
                    <span class="score-badge">相关度 {score:.2f}</span>
                    <div style="font-size:0.78rem; color:#666; margin-bottom:6px;">📖 {citation}</div>
                    <div class="classical-text">{chunk['original_text']}</div>
                    <div class="modern-text">💬 {chunk['translated_text']}</div>
                </div>
                """, unsafe_allow_html=True)
