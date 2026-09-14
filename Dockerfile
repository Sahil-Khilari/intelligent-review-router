# Matches .python-version and pyproject's requires-python = ">=3.13".
# The image previously used 3.11 while requirements.txt was resolved for 3.13.
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch: the default index pulls the CUDA wheels, which add several
# GB to an image that only ever runs inference on CPU.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

# Writable location for the SQLite database, mounted as a volume in compose.
RUN mkdir -p /data && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /data
USER appuser

ENV DATABASE_URL=sqlite:////data/ecommerce_router.db

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
