"""Project configuration and common paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CMU_RAW_DIR = RAW_DATA_DIR / "cmu"
CMU_RAW_FILE = CMU_RAW_DIR / "DSL-StrongPasswordData.csv"
CMU_FEATURES_FILE = PROCESSED_DATA_DIR / "cmu_features.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
USERS_DIR = PROJECT_ROOT / "users"

DATASET_REPORT_FILE = REPORTS_DIR / "dataset_report.md"

RANDOM_SEED = 42
