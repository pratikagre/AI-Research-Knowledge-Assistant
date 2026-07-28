import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Search home directory for .env first to resolve local credentials
home_env = Path.home() / ".env"
if home_env.exists():
    load_dotenv(home_env)

# Load current directory .env if present
load_dotenv()

class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "data" / "raw_documents"
    VECTOR_DB_DIR: Path = BASE_DIR / "data" / "vector_db"
    DATASET_DIR: Path = BASE_DIR / "data" / "dataset"
    MODELS_DIR: Path = BASE_DIR / "models"

    # Database
    DATABASE_URL: str = "sqlite:///data/assistant.db"

    # ML Classifier Model Path
    MODEL_PATH: Path = MODELS_DIR / "tf_classifier.h5"
    TOKENIZER_PATH: Path = MODELS_DIR / "tokenizer.pickle"

    # App Settings
    APP_NAME: str = "AI Research & Knowledge Assistant"
    DEBUG: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# Create directories if they do not exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
settings.DATASET_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
(settings.BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
