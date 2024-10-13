```python
import json
import requests

my_text = 'Развертывание SupaBase'
text = [my_text, my_text, my_text]
response = requests.post(
    url="http://embedder-reranker-kserve-service.embedder-reranker-service.svc.cluster.local:8080/v1/models/embedder:predict",
    json={'texts': text},
)
```


```python
import requests

query = ['машина']
contexts = ['вилка', 'автомобиль']


response = requests.post(
    url='http://embedder-reranker-kserve-service.embedder-reranker-service.svc.cluster.local:8080/v1/models/reranker:predict',
    json={
        'query': query,
        'contexts': contexts
    },
)
```