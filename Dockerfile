# 3.13 to match .python-version and pyproject's requires-python.
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages.
# requirements.txt pins torch==2.14.0+cu130, which is published only on the
# PyTorch CUDA index - PyPI has no such build, and a plain `pip install` fails
# with "No matching distribution found". The CUDA runtime libraries it needs
# (nvidia-*-cu13) come from PyPI as ordinary dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI backend with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
