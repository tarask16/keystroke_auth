from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
USERS_DIR = PROJECT_ROOT / "users"

RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

MLP_MODEL_PATH = MODELS_DIR / "mlp_baseline.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
