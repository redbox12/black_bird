from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    milvus_bs: int = 256
    embedder_reranker_host: str = "embedder-reranker:8080"
    filename: str = "/backend/app/data/dogovor.pdf"

_settings = Settings()