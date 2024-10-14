import logging

from typing import List, Dict
from pymilvus import (
    MilvusClient,
    FieldSchema,
    CollectionSchema,
    DataType
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

class CustomMilvusClient(MilvusClient):
    def __init__(
        self,
        milvus_endpoint: str,
        db_name: str
    ):
        super().__init__(
            uri=milvus_endpoint,
            db_name=db_name
        )

    def get_or_create_collection(self, collection_key: str) -> None:
        if self._get_connection().has_collection(collection_key):
            self.drop_collection(collection_key)
        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, descrition='Paragraph id', is_primary=True, auto_id=False),
            FieldSchema(name='content', dtype=DataType.JSON, descrition='Paragraph meta'),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='Embedding vectors', dim=1024)
        ]
        schema = CollectionSchema(fields=fields, description=f"{collection_key} collection")
        
        self.create_collection_with_schema(
            collection_name=collection_key,
            schema=schema,
            index_params = {
                "metric_type":"L2",
                "index_type":"IVF_FLAT",
                "params":{"nlist":1024}
            }
        )

    def insert_to_collection(self, collection_key: str, insert_data: List[Dict]) -> None:
        primary_keys = self.insert(collection_name=collection_key, data=insert_data)
        assert len(primary_keys) == len(insert_data), logger.error("Inserted less objects in collection that expected")
        
    def collection_query(self, collection_key: str, embedding: List[float]) -> List[int]:
        result_idxs = self.search(
            collection_name=collection_key,
            data=embedding,
            limit=10
        )[0]
        return [elem["id"] for elem in result_idxs]
