# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pip install open_clip_torch transformers torch torchvision qdrant-client kafka-python fastapi uvicorn[standard] opencv-python-headless Pillow numpy httpx
```
> Note: `requirements.txt` lists `openai-clip` (deprecated) — use `open_clip_torch` instead, as the Dockerfile does.

**Run the embedding service:**
```bash
python3 clip_service.py --config config.json
```

**Run the search API:**
```bash
uvicorn search_api:app --host 0.0.0.0 --port 8000
```

**Run the retention service** (deletes vectors older than 7 days, checks hourly):
```bash
python3 retention.py --config config.json
# Override defaults:
python3 retention.py --config config.json --retention-days 14 --interval-hours 6
```

**Build and run all services with Docker Compose:**
```bash
docker compose up --build -d        # build image and start all services
docker compose logs -f retention    # tail retention logs
docker compose down                 # stop everything
```

**Build and run a single container (legacy):**
```bash
docker build -t clip_service .
docker run --gpus all -v $(pwd)/config.json:/clip_service/config.json clip_service
```

## Architecture

This is a two-process vision-language search system for person re-identification across camera feeds.

### Data Flow

```
Kafka (person_crops topic)
    → clip_service.py  — consumes crops, encodes with CLIP, writes to Qdrant
    → Qdrant (persons collection)
    → search_api.py    — encodes text queries, searches Qdrant
    → ui/index.html    — search UI
    → Smart NVR        — snapshot proxy to avoid CORS
```

### clip_service.py — Indexing Service

Kafka consumer that batches person crop images (base64 JPEG), encodes them with OpenCLIP, and upserts vectors + metadata into Qdrant. Key parameters: `batch_size=32`, `batch_timeout_ms=200`. Uses exponential backoff on Kafka/Qdrant connection errors.

- `VisionEncoder`: wraps OpenCLIP model (`ViT-SO400M-14-SigLIP-384/webli`), exposes `encode_images()` and `encode_text()`
- `ensure_collection()`: creates the Qdrant `persons` collection with cosine distance and payload indexes on `sensor_id`, `tracker_id`, `pad_index`, `timestamp`
- `process_batch()`: decodes crops, generates embeddings, upserts `PointStruct` objects with full metadata
- Kafka consumer uses `auto_offset_reset="latest"` — only processes new messages after startup

### search_api.py — Query Service

FastAPI app that encodes natural-language text queries with the same CLIP model and performs vector search in Qdrant. Proxies NVR snapshot requests at `GET /snapshot` to avoid frontend CORS issues.

- `POST /search`: accepts `query`, `top_k`, `score_threshold`, optional `sensor_id`/`pad_index` filters
- `GET /snapshot`: proxies to Smart NVR at `nvr_base_url` from config
- `GET /health`: returns GPU/CPU device info
- `GET /stats`: returns Qdrant collection stats
- State (model, Qdrant client, HTTP client) initialized in `lifespan()` context manager and stored in module-level `_state` dict

### External Service Dependencies

All configured in `config.json`:
- **Kafka**: `bootstrap_servers`, `topic` (`person_crops`), `group_id`
- **Qdrant**: `host`/`port` (default `localhost:6333`), `collection_name` (`persons`)
- **Smart NVR**: `nvr_base_url` (default `http://localhost:8009`)

### ui/index.html

Single-file vanilla JS frontend. Calls `/search` and `/snapshot` on the same origin as the page. Renders results in a grid with bounding box overlays drawn on canvas using metadata from Qdrant payloads (supports both normalized 0–1 and absolute pixel coordinates).

## Known Issues

- `requirements.txt` references `openai-clip` (unmaintained); the correct package is `open_clip_torch`
- Both processes independently load the same CLIP model — expected by design for process isolation
- No test suite exists in this repository
