from transformers import AutoConfig, AutoTokenizer


class MlModels:
    def __init__(
        self,
        embedder_model_name,
        reranker_model_name,
    ):
        self._embedder_model_name = embedder_model_name
        self._reranker_model_name = reranker_model_name

        self._embedder_config = AutoConfig.from_pretrained(embedder_model_name)
        self._embedder_tokenizer = AutoTokenizer.from_pretrained(embedder_model_name)

        self._reranker_config = AutoConfig.from_pretrained(reranker_model_name)
        self._reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)

    def length_function(self, text):
        max_tokens = max(
            len(
                self._embedder_tokenizer.encode(
                    text,
                    truncation=False,
                    padding=False,
                    max_length=self._embedder_config.max_position_embeddings,
                )
            ),
            len(
                self._reranker_tokenizer.encode(
                    text,
                    truncation=False,
                    padding=False,
                    max_length=self._reranker_config.max_position_embeddings,
                )
            ),
        )
        return max_tokens


models = MlModels(
    "intfloat/multilingual-e5-large",
    "amberoad/bert-multilingual-passage-reranking-msmarco",
)
