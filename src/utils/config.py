"""
Path configuration for the Language Bottleneck Models project.

All paths are derived from two environment variables:
  - PROJECT_ROOT: Root of the repository (defaults to cwd).
  - DATA_DIR:     Root directory for external data assets
                  (e.g., Eedi, XES3G5M).  Defaults to PROJECT_ROOT/data/.

Set these in your shell or in a .env file before running any script.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env at repo root if present

# ── Core directories ──────────────────────────────────────────────────
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
PROJECT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ── XES3G5M dataset paths ────────────────────────────────────────────
XES3G5M_ROOT = os.environ.get("XES3G5M_ROOT", os.path.join(DATA_DIR, "XES3G5M"))
XES3G5M_PREPROCESSED = os.path.join(XES3G5M_ROOT, "preprocessed")
XES3G5M_TRAIN_PATH = os.path.join(XES3G5M_ROOT, "question_level/train_valid_sequences_quelevel.csv")
XES3G5M_TEST_PATH = os.path.join(XES3G5M_ROOT, "question_level/test_window_sequences_quelevel.csv")
XES3G5M_QUESTIONS_PATH = os.path.join(XES3G5M_ROOT, "metadata/questions.json")
XES3G5M_KC_MAP_PATH = os.path.join(XES3G5M_ROOT, "metadata/kc_routes_map.json")
XES3G5M_TRANSLATIONS_PATH = os.path.join(XES3G5M_ROOT, "translation/questions_translated_kc_sol_annotated_mapped.json")
