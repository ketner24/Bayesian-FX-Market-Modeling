FROM python:3.11-slim

WORKDIR /app

# System deps for PyMC/pytensor compilation + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libopenblas-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install IPOPT solver for Pyomo
RUN pip install --no-cache-dir ipopt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir .

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Create data directory for trade journal
RUN mkdir -p /app/data

# Prometheus metrics port + API port
EXPOSE 8000 9090

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "fx_engine.app:app", "--host", "0.0.0.0", "--port", "8000"]
