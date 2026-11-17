FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-download the all-MiniLM-L6-v2 embedding model at build time
RUN pip install --no-cache-dir sentence-transformers==2.2.2 && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY server/ ./server/
COPY rag/ ./rag/
COPY client/ ./client/

# Pre-download the embedding model one more time inside the app context
RUN python -c "from rag.retriever import HybridRetriever; print('Embedding model pre-downloaded')"

# Expose the port
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]