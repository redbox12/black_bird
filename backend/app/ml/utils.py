
import asyncio
import aiohttp
import logging
import json
import numpy as np
from typing import List
import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class InferenceEmbedder:
    def __init__(
        self,
        endpoint: str = "",
        batch_size: int = 4
    ) -> None:
        self._batch_size = batch_size
        self._endpoint = endpoint

    async def __create_chunks(self, objects: List) -> List:
        return [objects[i:i + self._batch_size] for i in range(0, len(objects), self._batch_size)]
    
    async def __process_batch(self, texts: List[str]) -> np.array:
        input_data_json = json.dumps({
            'texts': texts
        })
        async with aiohttp.ClientSession() as session:
            async with session.post(self._endpoint, data=input_data_json) as response:
                if response.status == 200:
                    result = await response.json()
                    return np.array(result["embeddings"])
                else:
                    response_text = await response.text()
                    raise Exception(f"InferenceEmbedderError: {response.status}, {response_text}")

    async def encode(self, texts: List[str]) -> np.array:
        texts = np.array([[item] for item in texts])
        chunks = await self.__create_chunks(texts)
        embeddings = await asyncio.gather(*(self.__process_batch(texts_batch) for texts_batch in chunks))
        return np.concatenate(embeddings)
    
class IndexerEmbedder:

    def __init__(
        self,
        endpoint: str
    ) -> None:

        self._embeddings_endpoint = endpoint


    def encode(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            url=self._embeddings_endpoint,
            json={"texts": texts},
        ).json()
        return response["embeddings"]

    def length_function(self, text: str) -> int:
        max_tokens = max(
            len(
                self._tokenizer.encode(
                    text,
                    truncation=False,
                    padding=False,
                    max_length=self._config.max_position_embeddings,
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
    
class InferenceReranker:
    def __init__(
        self,
        endpoint: str = "",
        batch_size: int = 64
    ) -> None:
        self._batch_size = batch_size
        self._endpoint = endpoint

    async def __process_batch(self, query: str, contexts: List[str]) -> np.array:
        input_data_json = json.dumps({
            'query': [query],
            'contexts': contexts
        })
        async with aiohttp.ClientSession() as session:
            async with session.post(self._endpoint, data=input_data_json) as response:
                if response.status == 200:
                    result = await response.json()
                    return np.array(result["scores"])
                else:
                    response_text = await response.text()
                    raise Exception(f"InferenceRerankerError: {response.status}, {response_text}")

    def __create_chunks(self, objects: List[str]) -> List[List[str]]:
        return [objects[i:i + self._batch_size] for i in range(0, len(objects), self._batch_size)]
    
    async def rerank(self, query: str, candidates: List[str]) -> np.array:
        chunks = self.__create_chunks(candidates)
        scores = await asyncio.gather(*(self.__process_batch(query, candidates_batch) for candidates_batch in chunks))
        return np.argsort(-np.concatenate(scores))
