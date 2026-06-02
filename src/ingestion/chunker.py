"""
Chunk 1: 文档解析与结构化切块
支持 TXT / PDF / DOC 三种格式的中医古籍
"""
import re
import json
from pathlib import Path
from typing import List, Dict
from loguru import logger


def parse_txt(file_path: str) -> List[Dict]:
    """解析纯文本格式的古籍，按段落和条文切块"""
    chunks = []
    book_name = Path(file_path).stem.split(".")[0]
    author = _infer_author(book_name)

    # 尝试多种编码读取文件
    encodings = ["utf-8", "gb18030", "gbk", "big5"]
    raw = ""
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                raw = f.read()
            break
        except UnicodeDecodeError:
            continue
            
    if not raw:
        logger.error(f"无法读取文本文件 {file_path}，所有编码尝试均失败")
        return []

    # 按换行符切分段落，过滤空白
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]

    current_chapter = "正文"
    article_counter = 0

    for para in paragraphs:
        # 检测章节标题（通常较短且不以句号结尾）
        if len(para) < 30 and not re.search(r"[。；：]$", para):
            current_chapter = para
            continue

        # 过滤太短的片段（少于10字）
        if len(para) < 10:
            continue

        article_counter += 1
        chunks.append({
            "chunk_id": f"{book_name}_{article_counter:04d}",
            "book": book_name,
            "author": author,
            "chapter": current_chapter,
            "article_id": str(article_counter),
            "original_text": para,
            "source_file": file_path,
        })

    logger.info(f"[TXT] {book_name}: 解析完成，共 {len(chunks)} 个 Chunk")
    return chunks


def parse_pdf(file_path: str) -> List[Dict]:
    """解析 PDF 格式的古籍"""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("请先安装 pypdf: pip install pypdf")
        return []

    chunks = []
    book_name = Path(file_path).stem
    author = _infer_author(book_name)
    reader = PdfReader(file_path)

    current_chapter = "正文"
    article_counter = 0
    buffer = ""

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # 将每页内容加入 buffer
        buffer += text + "\n"

    # 按段落切分整个文档
    paragraphs = [p.strip() for p in re.split(r"\n{2,}|　　", buffer) if p.strip()]

    for para in paragraphs:
        # 检测章节标题
        if len(para) < 40 and not re.search(r"[。；]", para) and re.search(r"[卷篇章节第]", para):
            current_chapter = para
            continue

        if len(para) < 15:
            continue

        article_counter += 1
        chunks.append({
            "chunk_id": f"{book_name}_{article_counter:04d}",
            "book": book_name,
            "author": author,
            "chapter": current_chapter,
            "article_id": str(article_counter),
            "original_text": para,
            "source_file": file_path,
        })

    logger.info(f"[PDF] {book_name}: 解析完成，共 {len(chunks)} 个 Chunk")
    return chunks


def parse_doc(file_path: str) -> List[Dict]:
    """解析 Word DOC/DOCX 格式的古籍"""
    try:
        from docx import Document
    except ImportError:
        logger.error("请先安装 python-docx: pip install python-docx")
        return []

    chunks = []
    book_name = Path(file_path).stem
    author = _infer_author(book_name)
    doc = Document(file_path)

    current_chapter = "正文"
    article_counter = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text or len(text) < 5:
            continue

        # 标题样式检测
        if para.style.name.startswith("Heading") or (len(text) < 40 and not re.search(r"[。；]$", text)):
            current_chapter = text
            continue

        article_counter += 1
        chunks.append({
            "chunk_id": f"{book_name}_{article_counter:04d}",
            "book": book_name,
            "author": author,
            "chapter": current_chapter,
            "article_id": str(article_counter),
            "original_text": text,
            "source_file": file_path,
        })

    logger.info(f"[DOC] {book_name}: 解析完成，共 {len(chunks)} 个 Chunk")
    return chunks


def parse_all_documents(data_dir: str) -> List[Dict]:
    """扫描 data 目录，自动路由解析所有支持格式的文件"""
    all_chunks = []
    data_path = Path(data_dir)

    for file in data_path.iterdir():
        suffix = file.suffix.lower()
        try:
            if suffix == ".txt":
                all_chunks.extend(parse_txt(str(file)))
            elif suffix == ".pdf":
                all_chunks.extend(parse_pdf(str(file)))
            elif suffix in (".doc", ".docx"):
                all_chunks.extend(parse_doc(str(file)))
            else:
                logger.warning(f"跳过不支持的文件格式: {file.name}")
        except Exception as e:
            logger.error(f"解析文件失败 {file.name}: {e}")

    logger.info(f"全部文档解析完成，共 {len(all_chunks)} 个 Chunk")
    return all_chunks


def _infer_author(book_name: str) -> str:
    """根据书名推断作者（简单规则映射）"""
    mapping = {
        "针灸甲乙经": "（晋）皇甫谧",
        "景岳全书": "（明）张景岳",
        "冷庐医话": "（清）陆以湉",
        "伤寒论": "（汉）张仲景",
        "黄帝内经": "（先秦）",
        "中风论": "待考",
    }
    for key, author in mapping.items():
        if key in book_name:
            return author
    return "佚名"


def save_chunks_to_jsonl(chunks: List[Dict], output_path: str):
    """将 chunks 保存为 JSONL 格式备用"""
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info(f"Chunks 已保存至: {output_path}")
