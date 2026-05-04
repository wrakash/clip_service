# CLIP Person Search Service

A vision-language search system for person re-identification across multiple camera feeds. It consumes person crop images from a Kafka topic, encodes them with a CLIP model (ViT-SO400M-14-SigLIP-384), stores the vectors in Qdrant, and exposes a natural-language search API with a browser UI.

## Architecture

```
Kafka (person_crops topic)
    → clip_service   — encodes crops with CLIP, writes vectors to Qdrant
    → Qdrant         — vector database (persisted via Docker volume)
    → search_api     — text query → CLIP encoding → Qdrant search → results
    → retention      — hourly cleanup of vectors older than 7 days
    → ui/index.html  — browser search interface (served by search_api at /ui)
```

## Prerequisites

Verify each before starting.

**1. Docker Engine**
```bash
docker --version        # 24.0 or later recommended
docker compose version  # v2.x required (not legacy docker-compose)
```
Install: https://docs.docker.com/engine/install/

**2. NVIDIA GPU drivers**
```bash
nvidia-smi              # must show your GPU and driver version
```
Install: https://www.nvidia.com/drivers

**3. NVIDIA Container Toolkit** (lets Docker access the GPU)
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```
If that fails, install the toolkit:
```bash
# Ubuntu / Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**4. Kafka broker** running and accessible (the `person_crops` topic must exist or be auto-created).

**5. Smart NVR** running at the URL you will set in `config.json` (for snapshot retrieval). The search and indexing features work without it; only "View Snapshot" in the UI requires it.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/clip_service.git
cd clip_service
```

### 2. Create your config file

```bash
cp config.example.json config.json
```

Open `config.json` and fill in your values:

```json
{
    "kafka": {
        "bootstrap_servers": "YOUR_KAFKA_HOST:9092",
        "topic": "person_crops",
        "group_id": "clip_embedding_service"
    },
    "qdrant": {
        "host": "localhost",
        "port": 6333,
        "collection_name": "persons"
    },
    "model": {
        "name": "ViT-SO400M-14-SigLIP-384",
        "pretrained": "webli"
    },
    "batch_size": 32,
    "batch_timeout_ms": 200,
    "nvr_base_url": "http://YOUR_NVR_HOST:PORT",
    "sensor_camera_map": {
        "YOUR_SENSOR_UUID_1": "entrance",
        "YOUR_SENSOR_UUID_2": "office-cam01"
    }
}
```

| Field | Description |
|---|---|
| `kafka.bootstrap_servers` | Kafka broker address(es) |
| `kafka.topic` | Topic that delivers base64-encoded person crop images |
| `qdrant.host` | Leave `localhost` — all services share the host network |
| `model.name` / `pretrained` | OpenCLIP model. Default is SigLIP-384 (best accuracy) |
| `batch_size` | Max crops to encode in a single GPU batch |
| `batch_timeout_ms` | Max wait time before flushing a partial batch |
| `nvr_base_url` | Base URL of your Smart NVR for snapshot retrieval |
| `sensor_camera_map` | Maps sensor UUIDs (from Kafka payloads) to human-readable camera names |

### 3. Build and start all services

```bash
docker compose up --build -d
```

This will:
- Pull the Qdrant image
- Build the application image (installs all Python dependencies)
- Download the CLIP model weights on first run (~2 GB from HuggingFace — takes a few minutes)
- Start all four services: `qdrant`, `clip_service`, `search_api`, `retention`

### 4. Verify everything is running

```bash
docker compose ps
```

Expected output — all services `Up`:
```
NAME                          STATUS
clip_service-qdrant-1         Up
clip_service-clip_service-1   Up
clip_service-search_api-1     Up
clip_service-retention-1      Up
```

Check the API:
```bash
curl http://localhost:8000/health
# {"status":"ok","device":"cuda"}

curl http://localhost:8000/stats
# {"vectors_count":0,"status":"green"}
```

`device` should be `cuda`. If it shows `cpu`, revisit the GPU prerequisites above.

---

## Usage

### Browser UI

Open in your browser:
```
http://SERVER_IP:8000/ui
```

Type a natural-language description (e.g. `"person wearing red jacket"`) and click **Search**. Click **View Snapshot** on any result to see the camera frame with bounding box overlay.

> **Score threshold:** CLIP text-to-image similarity scores are naturally low (0.1–0.2 range for this model). If you see no results, lower the **Min score** slider in the UI.

### Search API

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "person in blue shirt",
    "top_k": 10,
    "score_threshold": 0.1
  }'
```

**Filter by camera:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "person in blue shirt",
    "top_k": 10,
    "score_threshold": 0.1,
    "sensor_id": "YOUR_SENSOR_UUID"
  }'
```

### API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health and GPU status |
| `GET` | `/stats` | Total vectors indexed in Qdrant |
| `POST` | `/search` | Search by text description |
| `GET` | `/snapshot` | Proxy to NVR snapshot (used by UI) |
| `GET` | `/ui` | Browser search interface |

---

## Configuration

### Change data retention period

Edit the `retention` service command in `docker-compose.yml`:

```yaml
command: python3 retention.py --config config.json --retention-days 14 --interval-hours 1
```

Then apply:
```bash
docker compose up -d retention
```

### Useful commands

```bash
# View logs for a specific service
docker compose logs -f clip_service
docker compose logs -f search_api
docker compose logs -f retention

# Restart a single service
docker compose restart search_api

# Stop everything
docker compose down

# Stop everything and delete all stored vectors (irreversible)
docker compose down -v
```

---

## Troubleshooting

**`device: cpu` in /health — GPU not detected**
The NVIDIA Container Toolkit is not configured or the Docker daemon was not restarted after installation. Re-run `nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

**`clip_service` keeps restarting — Kafka connection errors**
Check `docker compose logs clip_service`. Verify `bootstrap_servers` in `config.json` is reachable from the host. The service retries automatically with a 15-second backoff.

**Search returns empty results**
Lower `score_threshold` to `0.05`–`0.1`. Also check `docker compose logs clip_service` to confirm vectors are being ingested (`Upserted N vectors` log lines).

**Snapshot returns 502**
The NVR at `nvr_base_url` is unreachable from the host. Check the URL and that the NVR service is running.

**Model download is slow or fails**
The SigLIP-384 model is ~2 GB. On first start, `clip_service` and `search_api` each download it independently. Subsequent starts use the Docker layer cache. If the download fails due to a rate limit, set a HuggingFace token:
```bash
# Add to docker-compose.yml under environment for clip_service and search_api:
environment:
  - HF_TOKEN=your_token_here
```
