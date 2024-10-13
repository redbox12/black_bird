from typing import Dict
import logging

import torch
import kserve
# from ray import serve
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
class EmbedderModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model = None
        self.gpu = torch.cuda.is_available()
        self.device = 'cuda' if self.gpu else 'cpu'
        self.load()

    def load(self) -> None:
        self.model = SentenceTransformer('intfloat/multilingual-e5-large', device=self.device)
        self.ready = True

    def predict(self, request_data: Dict, request_headers: Dict) -> Dict:
        texts = request_data['texts']
        # sentences = json.loads(base64.b64decode(request_data['instances'][0].encode('utf-8')).decode('utf-8'))
        result = self.model.encode(texts, normalize_embeddings=True).tolist()
        return {"embeddings": result}
    
class RerankModel(kserve.Model):
    def __init__(self, name: str, model_name):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model = None
        self.gpu = torch.cuda.is_available()
        self.device = 'cuda' if self.gpu else 'cpu'
        self.load(model_name)
        
    def load(self, model_name) -> None:
        # self.model = CrossEncoder('/home/jovyan/pakorolev/test/models/deep_pavlov_mrr_0_8413', max_length=512)
        self.model = CrossEncoder(model_name, max_length=512)
        self.ready = True

    def predict(self, request_data: Dict, request_headers: Dict) -> Dict:
        query = request_data['query'][0]
        contexts = request_data['contexts']
        instances = [[query, context] for context in contexts]
        logger.info(f"query: {query}\n\n\ncontexts: {contexts}\n\n\ninstances: {instances}")
        # instances = json.loads(base64.b64decode(request_data['instances'][0].encode('utf-8')).decode('utf-8'))
        # original_idxs = np.array(request_data['idxs'])
        scores = self.model.predict(instances, convert_to_numpy=True)
        if len(scores.shape) > 1:
                scores = scores[:,1]
        # scores_argsort = np.argsort(-scores).tolist()
        # return {"reranked_indexes": scores_argsort}
        return {"scores": scores.tolist()}


if __name__ == "__main__":
    embedder = EmbedderModel("embedder")
    reranker = RerankModel("reranker", 'amberoad/bert-multilingual-passage-reranking-msmarco')
    kserve.ModelServer(http_port=8080).start(models=[embedder, reranker])
