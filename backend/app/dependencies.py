from app.ml.utils import (
    InferenceReranker,
    InferenceEmbedder,
    IndexerEmbedder
)
from app.settings import _settings


def create_reranker_object() -> InferenceReranker:
    return InferenceReranker(
        endpoint=f"http://{_settings.embedder_reranker_host}/v1/models/reranker:predict",
        batch_size=64,
    )

def create_embedder_object() -> InferenceEmbedder :
    return InferenceEmbedder(
        endpoint=f"http://{_settings.embedder_reranker_host}/v1/models/embedder:predict",
        batch_size=4,
    )

def create_indexer_embedder_object() -> IndexerEmbedder:
    
    return IndexerEmbedder(
        endpoint=f"http://{_settings.embedder_reranker_host}/v1/models/embedder:predict"
    )