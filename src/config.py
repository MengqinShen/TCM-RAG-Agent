"""
TCM RAG System - Global Configuration
统一配置管理
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:latest")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

# ChromaDB
CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION_ORIGINAL = "tcm_original"
CHROMA_COLLECTION_TRANSLATED = "tcm_translated"

# Data
DATA_DIR = str(BASE_DIR / "data")

# Retrieval
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", 10))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", 5))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.3))

# Ontology
ONTOLOGY_PATH = str(BASE_DIR / "ontology" / "tcm_term_mapping.json")
