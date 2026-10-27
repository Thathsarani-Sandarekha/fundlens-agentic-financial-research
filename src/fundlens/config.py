"""Environment-based application configuration. Never commit real credentials."""
from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    app_debug: bool = False
    groq_api_key: SecretStr = SecretStr("")
    groq_model_default: str = "llama-3.3-70b-versatile"
    groq_model_fast: str = "llama-3.1-8b-instant"
    llm_timeout_seconds: float = 60.0

    # Deliberately no password embedded in application defaults.
    postgres_url: str = "postgresql+psycopg2://fundlens@localhost:5433/fundlens"
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = ""
    s3_secret_key: SecretStr = SecretStr("")
    s3_region: str = "us-east-1"
    s3_bucket: str = "fundlens-filings"
    neo4j_uri: str = "bolt://localhost:7688"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")

    data_raw_dir: Path = PROJECT_ROOT / "data/raw"
    data_processed_dir: Path = PROJECT_ROOT / "data/processed"

    def require_groq(self) -> None:
        if not self.groq_api_key.get_secret_value():
            raise RuntimeError("GROQ_API_KEY is not set. See .env.example")

settings = Settings()
