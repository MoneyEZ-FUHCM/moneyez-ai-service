FROM python:3.11-slim-buster

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        curl \
        tini \
        ffmpeg \
        libgomp1 \
        git \
        libsndfile1 \
        unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=/app \
    HF_HOME=/app/data/huggingface \
    HOME=/app \
    ENV=production

# Define build arguments for secrets
ARG GOOGLE_API_KEY

# Set environment variables from build arguments
ENV GOOGLE_API_KEY=$GOOGLE_API_KEY \
    QDRANT_PATH="/app/qdrant_data"

# Create non-root user
RUN addgroup --system appuser && \
    adduser --system --ingroup appuser appuser

# Install Python dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove gcc

# Copy the fixed data_stream.py to replace the problematic library file
COPY ./data_stream.py /usr/local/lib/python3.11/site-packages/assistant_stream/serialization/data_stream.py

# Copy application code
COPY --chown=appuser:appuser ./app ./app

RUN mkdir -p /app/qdrant_data && \
    chown -R appuser:appuser /app/qdrant_data && \
    chmod -R 777 /app/qdrant_data

# Expose port 8888 instead of 8080
EXPOSE 8888

USER appuser

ENTRYPOINT ["/usr/bin/tini", "--"]

# Use port 8888 in CMD
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8888", "--workers", "1", "--timeout-keep-alive", "300", "--log-level", "info"]
