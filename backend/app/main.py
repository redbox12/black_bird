import logging
import uvicorn

from langchain_community.retrievers import BM25Retriever
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.settings import _settings
from app.dependencies import create_reranker_object, create_embedder_object, create_indexer_embedder_object
from app.models import Item
from app.runners.simple_runners import IndexerRunner
from app.settings import _settings


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)



basic_embedder_object = create_embedder_object()
basic_reranker_object = create_reranker_object()
indexer_embedder_object = create_indexer_embedder_object()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bm25_retriever
    runner = IndexerRunner(embedder=basic_embedder_object, filename=_settings.filename)
    docs = await runner.run()
    bm25_retriever = BM25Retriever.from_documents(docs)




    yield


app = FastAPI(lifespan=lifespan)




@app.post("/question/")
async def create_item(item: Item):
    # search_engine = search_engines["doc_search"]
    # results = await asyncio.gather(*(search_engine.search(collection, item.question, SearchLevel.ONLY_SMART) for collection in search_engine.collections))
    # return {
    #     _settings.docs_collection_mapping[collection]: result
    #     for collection, result in zip(search_engine.collections, results)
    # }
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, port=_settings.port, host=_settings.host)