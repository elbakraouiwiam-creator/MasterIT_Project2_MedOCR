"""
Configuration settings for the Medication Recognition API
"""
from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # API
    API_TITLE: str = "Medication Box Recognition API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    PRODUCTS_DB_PATH: str = str(BASE_DIR / "data" / "produits.json")

    # OCR Settings
    OCR_LANGUAGES: list = ["fr", "ar", "en"]
    TESSERACT_CMD: str = "tesseract"

    # Matching
    FUZZY_MATCH_THRESHOLD: int = 60   # minimum score (0-100)
    TOP_N_RESULTS: int = 5

    # Image preprocessing
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

    class Config:
        env_file = ".env"


settings = Settings()
