import asyncio
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ultralytics import YOLO


MODEL_PATH = Path(__file__).with_name("yolov8n.pt")
RECORDINGS_DIR = Path(__file__).with_name("recordings")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ABSENCE_GRACE_SECONDS = 2
RECORDING_FPS = 5.0
RECORDING_MAX_WIDTH = 640
RECORDING_SEGMENT_SECONDS = 30

app = FastAPI()
telegram_executor = ThreadPoolExecutor(max_workers=1)
camera_in_use = False


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


def upload_telegram_file(method, field, video_path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"Telegram is not configured; recording retained at {video_path}")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    content_type = "video/mp4" if method == "sendVideo" else "application/octet-stream"
    try:
        with video_path.open("rb") as video_file:
            response = requests.post(
                url,
                data={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": "A person is no longer visible. Camera recording attached.",
                },
                files={field: (video_path.name, video_file, content_type)},
                timeout=(10, 90),
            )
        response.raise_for_status()
        result = response.json()
        if result.get("ok"):
            print(f"Telegram video sent: {video_path}")
            return True
        print(f"Telegram rejected {video_path.name}: {result.get('description')}")
    except Exception as error:
        print(f"Telegram upload failed for {video_path}: {error}")

    return False


def send_disappearance_alert(video_paths):
    for video_path in video_paths:
        if upload_telegram_file("sendVideo", "video", video_path):
            continue
        if upload_telegram_file("sendDocument", "document", video_path):
            continue
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                response = requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    json={
                        "chat_id": TELEGRAM_CHAT_ID,
                        "text": f"Could not upload the recording. It is saved locally at: {video_path}",
                    },
                    timeout=(10, 30),
                )
                response.raise_for_status()
            except Exception as error:
                print(f"Telegram notification failed: {error}")


def recording_frame(frame):
    height, width = frame.shape[:2]
    if width > RECORDING_MAX_WIDTH:
        height = int(height * RECORDING_MAX_WIDTH / width)
        frame = cv2.resize(frame, (RECORDING_MAX_WIDTH, height))
    height, width = frame.shape[:2]
    return frame[: height - height % 2, : width - width % 2]


def start_recording(state, frame, now):
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = RECORDINGS_DIR / f"human_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
    output_frame = recording_frame(frame)
    height, width = output_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        RECORDING_FPS,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        print("Could not start MP4 recording.")
        return

    state["writer"] = writer
    state["path"] = path
    state["segment_started_at"] = now


def finish_recording(state):
    writer = state.get("writer")
    path = state.get("path")
    if writer is None:
        return

    writer.release()
    if path and path.exists() and path.stat().st_size:
        state["completed_paths"].append(path)
    state["writer"] = None
    state["path"] = None
    state["segment_started_at"] = None


def update_recording(state, frame, person_detected, now):
    if person_detected:
        state["last_detection_time"] = now
        if state["writer"] is None:
            start_recording(state, frame, now)
    elif (
        state["writer"] is not None
        and now - state["last_detection_time"] > ABSENCE_GRACE_SECONDS
    ):
        finish_recording(state)
        if state["completed_paths"]:
            telegram_executor.submit(
                send_disappearance_alert, tuple(state["completed_paths"])
            )
            state["completed_paths"].clear()

    if state["writer"] is not None:
        state["writer"].write(recording_frame(frame))
        if now - state["segment_started_at"] >= RECORDING_SEGMENT_SECONDS:
            finish_recording(state)
            if person_detected:
                start_recording(state, frame, now)


def detect_and_encode(model, frame, state):
    result = model.predict(
        source=frame,
        classes=[0],
        imgsz=416,
        max_det=10,
        conf=0.35,
        verbose=False,
    )[0]
    count = len(result.boxes)
    now = time.monotonic()
    annotated = result.plot()
    update_recording(state, annotated, count > 0, now)
    success, encoded = cv2.imencode(
        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75]
    )
    if not success:
        raise RuntimeError("Could not encode the camera frame")
    return {
        "image": base64.b64encode(encoded).decode("ascii"),
        "count": count,
        "recording": state["writer"] is not None,
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Human Detector</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; background: #101614; color: #eef4ef; }
body { max-width: 680px; margin: 0 auto; padding: 20px 16px 36px; }
h1 { margin: 4px 0 8px; font-size: 1.5rem; }
p { color: #b6c5bc; }
.view { position: relative; width: 100%; aspect-ratio: 4 / 3; overflow: hidden; background: #25312c; border-radius: 8px; }
video, #result { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: fill; }
#result { display: none; }
.controls { display: flex; gap: 10px; margin: 14px 0; }
button { min-height: 48px; flex: 1; border: 0; border-radius: 6px; font: inherit; font-weight: 650; }
#start { background: #baf36b; color: #172313; }
#stop { background: #394740; color: #fff; }
#status { min-height: 1.5em; margin-top: 14px; color: #d8e6dc; }
#recording { color: #ff8a72; font-weight: 700; }
</style>
</head>
<body>
<h1>Human Detector</h1>
<p>Live person detection with local recording and optional Telegram alerts.</p>
<div class="view" id="view">
  <video id="camera" autoplay playsinline muted></video>
  <img id="result" alt="Live detection result">
</div>
<div class="controls">
  <button id="start">Start camera</button>
  <button id="stop" disabled>Stop</button>
</div>
<div id="status" role="status">Camera is off.</div>
<script>
const camera = document.querySelector('#camera');
const result = document.querySelector('#result');
const view = document.querySelector('#view');
const status = document.querySelector('#status');
const startButton = document.querySelector('#start');
const stopButton = document.querySelector('#stop');
const canvas = document.createElement('canvas');
const context = canvas.getContext('2d');
let stream;
let socket;

function stopCamera() {
  if (socket) socket.close();
  if (stream) stream.getTracks().forEach(track => track.stop());
  socket = null;
  stream = null;
  camera.srcObject = null;
  result.style.display = 'none';
  startButton.disabled = false;
  stopButton.disabled = true;
  status.textContent = 'Camera is off.';
}

function captureFrame() {
  if (!socket || socket.readyState !== WebSocket.OPEN || camera.readyState < 2) return;
  const scale = Math.min(1, 640 / camera.videoWidth);
  canvas.width = Math.round(camera.videoWidth * scale);
  canvas.height = Math.round(camera.videoHeight * scale);
  view.style.aspectRatio = `${canvas.width} / ${canvas.height}`;
  context.drawImage(camera, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(blob => {
    if (blob && socket && socket.readyState === WebSocket.OPEN) socket.send(blob);
  }, 'image/jpeg', 0.72);
}

startButton.addEventListener('click', async () => {
  startButton.disabled = true;
  status.textContent = 'Requesting camera access...';
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 640 } }
    });
    camera.srcObject = stream;
    await camera.play();
    const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${scheme}//${location.host}/stream`);
    socket.onopen = () => { stopButton.disabled = false; status.textContent = 'Loading model...'; };
    socket.onmessage = event => {
      const message = JSON.parse(event.data);
      if (message.error) {
        status.textContent = message.error;
        stopCamera();
      } else if (message.ready) {
        status.textContent = 'Model ready. Detecting...';
        captureFrame();
      } else {
        result.src = `data:image/jpeg;base64,${message.image}`;
        result.style.display = 'block';
        status.innerHTML = message.count
          ? `Person detected: ${message.count}${message.recording ? ' · <span id="recording">Recording</span>' : ''}`
          : `No person detected${message.recording ? ' · <span id="recording">Recording</span>' : ''}`;
        captureFrame();
      }
    };
    socket.onerror = () => { status.textContent = 'Camera connection failed.'; };
    socket.onclose = () => { if (stream) stopCamera(); };
  } catch (error) {
    status.textContent = `Could not start camera: ${error.message}`;
    stopCamera();
  }
});

stopButton.addEventListener('click', stopCamera);
</script>
</body>
</html>"""
    )


@app.websocket("/stream")
async def stream_camera(websocket: WebSocket):
    global camera_in_use
    if camera_in_use:
        await websocket.close(code=1013, reason="Camera is already in use")
        return

    camera_in_use = True
    state = {
        "writer": None,
        "path": None,
        "segment_started_at": None,
        "last_detection_time": 0.0,
        "completed_paths": [],
    }
    try:
        await websocket.accept()
        try:
            model = await asyncio.to_thread(load_model)
        except Exception as error:
            await websocket.send_text(json.dumps({"error": f"Model load failed: {error}"}))
            return

        await websocket.send_text(json.dumps({"ready": True}))
        while True:
            image_bytes = await websocket.receive_bytes()
            frame = await asyncio.to_thread(
                cv2.imdecode,
                np.frombuffer(image_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is None:
                continue
            response = await asyncio.to_thread(detect_and_encode, model, frame, state)
            await websocket.send_text(json.dumps(response))
    except WebSocketDisconnect:
        pass
    except Exception as error:
        print(f"Camera stream failed: {error}")
        try:
            await websocket.send_text(json.dumps({"error": f"Camera processing failed: {error}"}))
        except Exception:
            pass
    finally:
        finish_recording(state)
        camera_in_use = False