# roomi-backend — Python 3.11 + CUDA (via NVIDIA Container Toolkit on host)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libs needed by OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install torch with CUDA first (cu121 = RTX 3050 + CUDA 12.1)
RUN pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121

# Install the rest
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p app/static/precached app/static/results

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
