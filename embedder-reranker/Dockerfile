FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY src/serve.py serve.py

RUN pip install kserve==0.11.1 sentence-transformers==2.2.2
CMD python serve.py
EXPOSE 8080
