FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/opt/models/tiktoken \
    FASTEMBED_CACHE_PATH=/opt/models/fastembed

# tesseract: OCR for documents without a text layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt -r requirements-dev.txt

# Bake model files into the image so nothing is downloaded at runtime.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/models/fastembed')" \
    || echo "WARNING: embedding model not cached; the service will fall back to BM25-only retrieval"
ENV HF_HUB_OFFLINE=1

COPY app ./app
COPY tests ./tests

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=60 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
