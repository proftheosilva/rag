import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = "openrouter/auto"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    # Configuração da origem de recebimento dos documentos ('raw' ou 'webhook')
    DOCUMENT_SOURCE: str = os.getenv("DOCUMENT_SOURCE", "raw")
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")
    WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")

    # Diretórios da Arquitetura Medalhão e DuckDB
    # No container docker ./dados/documentos é montada em /app/documentos.
    # No ambiente de host (Windows), fica em .../dados/documentos.
    _default_base = (
        "/app/documentos"
        if os.path.exists("/app/documentos")
        else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "dados", "documentos"))
    )
    BASE_DOCS_DIR: str = os.getenv("BASE_DOCS_DIR", _default_base)
    RAW_DIR: str = os.getenv("RAW_DIR", os.path.join(BASE_DOCS_DIR, "raw"))
    SILVER_DIR: str = os.getenv("SILVER_DIR", os.path.join(BASE_DOCS_DIR, "silver"))
    GOLD_DIR: str = os.getenv("GOLD_DIR", os.path.join(BASE_DOCS_DIR, "gold"))
    DATABASE_DIR: str = os.getenv("DATABASE_DIR", os.path.join(BASE_DOCS_DIR, "database"))
    DUCKDB_PATH: str = os.getenv("DUCKDB_PATH", os.path.join(DATABASE_DIR, "analytics.duckdb"))

    # Mantido para retrocompatibilidade: aponta para RAW_DIR
    DOCS_DIR: str = RAW_DIR

    @property
    def is_webhook_source(self) -> bool:
        return self.DOCUMENT_SOURCE.lower() == "webhook"

    @property
    def is_raw_source(self) -> bool:
        return self.DOCUMENT_SOURCE.lower() == "raw"

settings = Settings()

