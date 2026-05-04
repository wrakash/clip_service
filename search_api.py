"""
Person Search API
=================
FastAPI service that accepts a natural-language text query, encodes it with
CLIP, and returns the top-K matching person crops from Qdrant.  Includes a
snapshot proxy to the Smart NVR so the frontend can fetch camera frames
without CORS issues.

Run:
    uvicorn search_api:app --host 0.0.0.0 --port 8000
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import httpx
import numpy as np
# dummy imports
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("search_api")

COLLECTION_NAME = "persons"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("config.json") as f:
        config = json.load(f)

    device = "cpu"
    model = None
    tokenizer = None

    qdrant = QdrantClient(
        host=config["qdrant"]["host"],
        port=config["qdrant"]["port"],
    )

    _state["model"]     = model
    _state["tokenizer"] = tokenizer
    _state["device"]    = device
    _state["qdrant"] = qdrant
    _state["config"] = config
    _state["sensor_camera_map"] = config.get("sensor_camera_map", {})
    _state["nvr_base_url"] = config.get("nvr_base_url", "http://localhost:8009")
    _state["http_client"] = httpx.AsyncClient(timeout=15.0)

    logger.info("Search API ready")
    yield

    await _state["http_client"].aclose()
    _state.clear()


app = FastAPI(title="Person Search API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# CLIP text encoding
# ---------------------------------------------------------------------------

def _encode_text(text: str) -> List[float]:
    import numpy as np
    # dummy random embedding
    vec = np.random.randn(512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query:           str
    top_k:           int   = 10
    score_threshold: float = 0.05
    sensor_id:  Optional[str] = None
    pad_index:  Optional[int] = None


class PersonResult(BaseModel):
    id:           str
    score:        float
    sensor_id:    str
    camera_name:  str
    tracker_id:   int
    timestamp:    str
    bbox:         List[float]
    frame_number: int
    confidence:   float
    pad_index:    int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/search", response_model=List[PersonResult])
def search(req: SearchRequest):
    """Search for persons matching a natural-language description."""
    query_vector = _encode_text(req.query)

    must = []
    if req.sensor_id:
        must.append(FieldCondition(key="sensor_id", match=MatchValue(value=req.sensor_id)))
    if req.pad_index is not None:
        must.append(FieldCondition(key="pad_index", match=MatchValue(value=req.pad_index)))

    qdrant_filter = Filter(must=must) if must else None

    try:
        response = _state["qdrant"].query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=req.top_k,
            query_filter=qdrant_filter,
            score_threshold=req.score_threshold,
        )
        hits = response.points
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sensor_map = _state["sensor_camera_map"]
    results = []
    for hit in hits:
        p = hit.payload
        sid = p.get("sensor_id", "")
        results.append(PersonResult(
            id=str(hit.id),
            score=round(hit.score, 4),
            sensor_id=sid,
            camera_name=sensor_map.get(sid, sid),
            tracker_id=p.get("tracker_id", -1),
            timestamp=p.get("timestamp", ""),
            bbox=p.get("bbox", []),
            frame_number=p.get("frame_number", 0),
            confidence=p.get("confidence", 0.0),
            pad_index=p.get("pad_index", 0),
        ))

    return results


@app.get("/snapshot")
async def snapshot_proxy(
    camera: str = Query(..., description="NVR camera name"),
    timestamp: str = Query(..., description="ISO 8601 timestamp"),
    quality: int = Query(85, ge=1, le=95),
):
    """
    Proxy to Smart NVR /snapshot endpoint.  The frontend calls this to avoid
    CORS issues.  The NVR auto-cleans the JPEG file after serving.
    """
    nvr = _state["nvr_base_url"]
    url = f"{nvr}/snapshot"
    params = {"camera": camera, "timestamp": timestamp, "quality": quality}

    try:
        resp = await _state["http_client"].get(url, params=params)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"NVR unreachable: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=resp.text[:500],
        )

    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "image/jpeg"),
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": _state.get("device", "unknown")}


@app.get("/stats")
def stats():
    try:
        info = _state["qdrant"].get_collection(COLLECTION_NAME)
        return {
            "vectors_count":         info.points_count,
            "status":                str(info.status),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Static files — serve the frontend UI
# ---------------------------------------------------------------------------

ui_path = Path(__file__).parent / "ui"
if ui_path.is_dir():
    app.mount("/ui", StaticFiles(directory=str(ui_path), html=True), name="ui")
