import cv2
import base64
import json
import time
import logging
import threading
from datetime import datetime, timezone
from kafka import KafkaProducer
from ultralytics import YOLO
from flask import Flask, request, Response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Snapshot Server Logic ---
app = Flask(__name__)
frame_cache = {} # Cache for snapshots: {timestamp: frame}
cache_lock = threading.Lock()
MAX_CACHE_SIZE = 100 # Store last 100 frames

@app.route('/snapshot')
def get_snapshot():
    camera = request.args.get('camera')
    timestamp = request.args.get('timestamp')
    
    with cache_lock:
        frame = frame_cache.get(timestamp)
    
    if frame is not None:
        _, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    else:
        return "Snapshot not found", 404

def start_snapshot_server(port=8009):
    logger.info(f"Starting snapshot server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# --- Streamer Logic ---

def run_video_streamer(video_source="video.mp4", kafka_bootstrap='localhost:9092', sensor_id="office-cam01"):
    # Start snapshot server in background
    server_thread = threading.Thread(target=start_snapshot_server, daemon=True)
    server_thread.start()

    # 1. Load YOLOv8 model
    logger.info("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    # 2. Setup Kafka Producer
    logger.info(f"Connecting to Kafka at {kafka_bootstrap}...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=[kafka_bootstrap],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            retries=5
        )
    except Exception as e:
        logger.error(f"Failed to connect to Kafka: {e}")
        return

    # 3. Open Video Source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Could not open video source: {video_source}")
        if video_source != 0:
            logger.info("Attempting to open webcam (0) as fallback...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return

    logger.info(f"Starting processing. Press Ctrl+C to stop.")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream or error reading frame.")
                break

            frame_count += 1
            ts = datetime.now(timezone.utc).isoformat()
            
            # Run inference
            results = model(frame, verbose=False)

            detected_person = False
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls == 0:  # Person
                        detected_person = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0: continue

                        _, buffer = cv2.imencode('.jpg', crop)
                        img_b64 = base64.b64encode(buffer).decode('utf-8')

                        message = {
                            "sensor_id": sensor_id,
                            "tracker_id": 1,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                            "frame_number": frame_count,
                            "pad_index": 0,
                            "crop_jpeg_b64": img_b64,
                            "timestamp": ts
                        }
                        producer.send("person_crops", message)

            if detected_person:
                with cache_lock:
                    frame_cache[ts] = frame.copy()
                    if len(frame_cache) > MAX_CACHE_SIZE:
                        oldest_ts = next(iter(frame_cache))
                        del frame_cache[oldest_ts]

            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames...")
                producer.flush()

    except KeyboardInterrupt:
        logger.info("Stopping streamer...")
    finally:
        cap.release()
        producer.flush()
        producer.close()
        logger.info("Resources released.")

if __name__ == "__main__":
    run_video_streamer(video_source=0, sensor_id="office_cam_01")
