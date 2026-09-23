from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
LLM_BOOK_PATH = ASSETS_DIR / "llm-book.txt"
SPM_MODEL_PREFIX = ASSETS_DIR / "llm_book_spm"