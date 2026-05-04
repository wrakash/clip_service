FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /clip_service

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    open_clip_torch \
    transformers \
    qdrant-client \
    kafka-python \
    fastapi \
    "uvicorn[standard]" \
    opencv-python-headless \
    Pillow \
    numpy \
    httpx

COPY . /clip_service

CMD ["python3", "clip_service.py", "--config", "config.json"]
