from app.storage.milvus import CustomMilvusClient
from app.settings import _settings
from app.ml.utils import (
    IndexerEmbedder
)

from joblib import Parallel, delayed, cpu_count
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
import logging
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from langchain.docstore.document import Document
import math
from joblib import Parallel, delayed, cpu_count

docs_url = "https://cdn.cloud.ru/docs_index.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger()


class Models:
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


class IndexerRunner:
    def __init__(self, embedder: IndexerEmbedder, filename: str):
        self.milvus_client = CustomMilvusClient(
            milvus_endpoint="http://standalone:19530",
            db_name="default"
        )
        self.embedder = embedder
        self.filename = filename
        self.models = Models(
            "intfloat/multilingual-e5-large",
            "amberoad/bert-multilingual-passage-reranking-msmarco",
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, length_function=self.models.length_function
        )
    
    def format_document(self, doc):
        return Document(
            page_content=doc['content'],
            metadata={
                "chapter": doc["chapter"],
            },
        )
    
    def batch_docs(self, docs, n_jobs):
            batch_size = math.ceil(len(docs) / n_jobs)
            return [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]
    
    def process_batch(self, batch):
            return self.text_splitter.split_documents(batch)
    
    def split_docs(self):
        loader = PyPDFLoader(self.filename)
        pages = loader.load()
        full_text = ''.join([page.page_content for page in pages])
        split_text = full_text.split("\nРаздел ")
        split_text = [f"Раздел {fragment.strip()}" for fragment in split_text if fragment.strip()]
        documents = [{"content": content, "chapter": n + 1} for n, content in enumerate(split_text[1:])]
        n_jobs = -1
        formatted_docs = Parallel(n_jobs=n_jobs)(
            delayed(self.format_document)(doc) for doc in tqdm(documents)
        )
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        batched_docs = self.batch_docs(formatted_docs, n_jobs)
        docs = Parallel(n_jobs=n_jobs)(
            delayed(self.process_batch)(batch) for batch in tqdm(batched_docs)
        )
        docs = [doc for document in docs for doc in document]
        return docs
    
    def upload_to_milvus(self, collection_data) -> None:
        self.milvus_client.get_or_create_collection(collection_key="dogovor")
        for batch_idx in tqdm(
            list(range((len(collection_data) // _settings.milvus_bs) + 1))
        ):
            batch_data = collection_data[
                batch_idx * _settings.milvus_bs : (batch_idx + 1) * _settings.milvus_bs
            ]
            if len(batch_data) == 0:
                continue
            batch_embeddings = self.embedder.encode(
                [elem.page_content for elem in batch_data]
            )
            # TODO: Think about hardcoded keys, remove content insertion to milvus (it's for debug)
            insert_data = [
                {"id": idx, "content": {"context":elem.page_content, "chapter":elem.metadata["chapter"]}, "embedding": embedding}
                for idx, elem, embedding in zip(
                    list(
                        range(
                            batch_idx * _settings.milvus_bs,
                            (batch_idx + 1) * _settings.milvus_bs,
                        )
                    ),
                    batch_data,
                    batch_embeddings,
                )
            ]
            self.milvus_client.insert_to_collection(
                collection_key="dogovor", insert_data=insert_data
            )
            return True


    def run(self):
        docs = self.split_docs()
        _ = self.upload_to_milvus(docs)
        return docs