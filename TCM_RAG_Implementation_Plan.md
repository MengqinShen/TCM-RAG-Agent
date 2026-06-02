# 中医古籍智能问答与辅助诊疗系统 - 实施计划 (Implementation Plan)

本文档基于业务需求文档（BRD）中的“混合 RAG + 知识图谱 + 本体桥接”方案，提供端到端的代码级落地实施计划。此文档将作为后续编码和项目开发的核心参考。

## 一、 系统架构与技术栈选型

系统必须满足**完全私有化部署**的安全底线，因此所有组件均需在本地服务器或私有云运行。

*   **基础服务层 (Infrastructure):** Docker & Docker Compose
*   **LLM 推理引擎 (LLM Serving):** vLLM 或 Ollama (用于挂载神农中医药大模型 ShenNong-TCM-LLM 或 Alpaca-Plus-7B)
*   **向量数据库 (Vector DB):** Milvus 或 Qdrant (用于存储双路 Embedding 向量)
*   **全文/关键字搜索引擎 (Lexical DB):** Elasticsearch (用于 BM25 检索及准确术语匹配)
*   **知识图谱/本体层 (Graph/Ontology DB):** Neo4j (用于存储古今术语对照词典与实体关系)，轻量级可使用本地 JSON/SQLite
*   **RAG 编排框架 (RAG Framework):** V0 MVP 阶段优先使用 LangChain 快速验证基础直线型问答链路；后续迭代优化时升级为 LangGraph，以支持复杂的多智能体与鉴别诊断状态流转。
*   **后端 API 服务 (Backend Framework):** Python + FastAPI
*   **向量与重排模型:** BGE-M3 (双语/古文支持较好), BGE-Reranker-V2-M3

---

## 二、 第一层：知识摄取管道（离线-构建期）

**核心目标：** 解决文言文与白话文鸿沟，构建带丰富元数据的双路索引。

### 1. 语料清洗与结构化分段 (Chunking)
*   **任务：** 接入《黄帝内经》、《伤寒论》等 TXT/PDF 原始文件。
*   **实现：** 编写 Python 脚本，基于古籍特有的“卷-篇-章-条文”结构进行层级解析切块。
*   **输出格式 (JSONL)：** `{"book": "伤寒论", "chapter": "辨太阳病脉证并治", "article_id": "1", "original_text": "太阳之为病，脉浮，头项强痛而恶寒。"}`

### 2. LLM 自动化白话文翻译与注释 (Translation Pipeline)
*   **任务：** 调用本地部署的神农大模型，对每一条古文进行精准的现代临床白话文翻译。
*   **实现：** 使用 `asyncio` + `vLLM API` 提升批处理速度。Prompt 需要严格限定输出格式为纯白话文解释。

### 3. 医学命名实体识别 (NER) 与古今术语对照 (Ontology Bridge)
*   **任务：** 从原文和白话文中提取“症候”、“疾病”、“药材”、“治法”、“穴位”。
*   **实现：** 
    *   借助开源医疗 NLP 库 (如 HanLP/Spacy) 或通过 LLM Few-shot 提取。
    *   构建映射字典：例如 `{"消渴": ["ICD-10 E14", "糖尿病"], "头项强痛": ["颈部僵硬", "头痛"]}`。

### 4. 双路嵌入与数据入库 (Dual-Embedding Ingestion)
*   **任务：** 对文本进行向量化并存入数据库。
*   **实现：**
    *   调用 `BGE-M3` 将 `original_text` 和 `translated_text` 分别向量化。
    *   **核心逻辑：** `original_text` 和 `translated_text` 在逻辑上是**绝对配对 (Paired)** 的。在向量数据库中，它们同属于一个数据块 (Chunk) 的两个属性。
    *   **Milvus/Qdrant 插入：** 采用“双向量场 (Multi-Vector)”结构，分别对 `original_text` 和 `translated_text` 生成向量，但它们指向同一个实体 (Entity ID)。
    *   **Metadata 处理：** 原始的古文 (`original_text`) 以及白话文翻译 (`translated_text`) 都会作为文本格式存储在 Metadata（元数据/Payload）中。当系统通过白话文向量命中该区块时，可以直接从 Metadata 中把古文原文提取出来，喂给下游生成层的 LLM。

---

## 三、 第二层：查询处理管道（在线-实时）

**核心目标：** 精准理解用户大白话意图，利用多路召回最大化相关性。

### 1. 查询意图分析与 NER (Query Processing)
*   **任务：** 对用户输入的现代白话文提问（如：“患者一直口渴想喝水，尿多，吃得多但消瘦，怎么办？”）进行解析。
*   **实现：** 在线调用轻量级 LLM/NER 服务，提取症状实体：`["口渴", "多饮", "多尿", "多食", "消瘦"]`。

### 2. 查询扩展与本体桥接 (Query Expansion)
*   **任务：** 消除古今术语鸿沟。
*   **实现：** 将提取出的现代症状实体放入【古今术语对照词典】(Neo4j/SQLite) 进行查询，将其翻译扩展为古籍术语。例如，扩展查询词为：`"口渴 多尿 消瘦 (消渴)"`。

### 3. 混合检索与知识图谱遍历 (Hybrid Retrieval)
*   **任务：** 从海量古籍中召回高相关性条文。
*   **实现：**
    *   *Path A (Vector Search):* 用扩展后的白话文向量在 Milvus 中进行 Cosine Similarity 检索（召回 Top-10）。
    *   *Path B (BM25 Search):* 用古籍术语在 Elasticsearch 中进行关键词 Exact Match（召回 Top-10）。
    *   将两路召回结果合并去重。

### 4. 交叉编码器重排序 (Cross-Encoder Reranking)
*   **任务：** 解决向量检索导致的“语义相似但医学逻辑不符”问题。
*   **实现：** 将用户的原始 Query 与合并召回的 20 条古文+白话文 Chunk 组成 pairs，送入 `BGE-Reranker-V2` 模型重新打分，提取真正的 Top-5 高质量证据段落。

---

## 四、 第三层：回答生成与幻觉控制（在线-实时）

**核心目标：** 生成专业、安全、有据可查的医疗答复。

### 1. 鉴别诊断链式思维 (CoT Prompt Engineering)
*   **任务：** 组装 Final Prompt 给主力 LLM（ShenNong-TCM-LLM）。
*   **实现：** 
    构建类似如下的 System Prompt：
    ```text
    你是一位严谨的中医临床专家。请根据以下[参考古籍原文及翻译]，回答患者的提问。
    要求：
    1. 必须使用现代临床医学语言解释。
    2. 每一条病理分析和治疗方案（包含方剂），必须引用给定的参考资料。
    3. 引用格式强制为：【书名·篇章，条文编号】。
    
    [参考古籍原文及翻译]：{retrieved_chunks_with_metadata}
    [用户问题]：{user_query}
    ```

### 2. 阈值拦截与拒答机制 (Confidence Thresholding & Safety Guardrail)
*   **任务：** 医疗准确性是安全底线，防止无中生有。
*   **实现：** 
    *   如果在重排序 (Reranking) 阶段，Top-1 段落的得分低于预设的安全阈值（例如 score < 0.3），拦截大模型生成。
    *   系统直接返回硬编码的兜底文案：“【文献中未找到相关记载】，为确保医疗安全，无法提供无据诊断，请重新描述症状或查阅其他资料。”

---

## 五、 第四层：前端交互与 UI 层 (Frontend UI & Interaction)

**核心目标：** 建立医生强信任感，提供透明溯源与高效输入的交互界面。

### 1. 核心布局：双分屏对照视图 (Split-Screen UI)
*   **左侧（主对话区）：** 展示医生的大白话提问，以及 AI 用现代医学语言生成的“辩证思路”和“方剂推荐”。
*   **右侧（溯源与古文验证区）：** 当左侧生成内容带有 `[1]`, `[2]` 等引用标记时，右侧面板同步高亮显示被命中的古文原文、出处及翻译。医生点击引用标号，右侧自动滚动至原典证据。

### 2. 高效输入：结构化病历与自然语言结合
*   提供“望闻问切”快捷结构化表单（主诉、舌象、脉象等），减少医生打字成本，并方便后端构建高质量 Prompt。
*   支持直接复制粘贴 EMR（电子病历）记录，由 AI 自动提取症状实体。

### 3. 特殊状态提示与反馈机制 (Human-in-the-Loop)
*   **低置信度警告：** 触发后端拒答时，弹出醒目的 Alert 提示：“未匹配到高置信度古籍原文，系统停止生成诊断建议”。
*   **一键反馈：** 提供“👍 采纳”、“👎 查无此文”、“⚠️ 翻译有误”等按钮，记录数据用于后续检索与大模型的调优。

### 4. 技术栈选型
*   **V0 原型期：** 采用 Streamlit 或 Gradio，利用 Markdown 分栏快速实现“左侧对话，右侧看原文”，极速验证业务流程。
*   **V1 生产期：** 升级至 React (Next.js) + TailwindCSS 或 Vue3，提供更平滑的高亮交互及与医院 HIS/EMR 系统的嵌入整合。

---

## 六、 开发阶段划分与里程碑 (Milestones)

1.  **Milestone 1 (数据与基础设施):** 跑通《伤寒论》单本书的清洗、离线白话文翻译、NER 提取及向量化入库（Elasticsearch + Qdrant）。
2.  **Milestone 2 (检索与召回中台):** 实现 Query 解析、本体词典扩展、以及双路混合检索与 Reranker 重排序逻辑封装。
3.  **Milestone 3 (Agent与生成):** 部署 ShenNong-TCM-LLM，编写 CoT 提示词，实现符合引用规范的回答生成及低阈值拒答逻辑。
4.  **Milestone 4 (API与全链路):** 封装 FastAPI 接口，实现 `POST /api/v1/query`，端到端连通。
