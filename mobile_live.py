import asyncio
import base64
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ultralytics import YOLO


MODEL_PATH = Path(__file__).with_name("yolov8n.pt")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ABSENCE_GRACE_SECONDS = 2
PERSON_INFERENCE_INTERVAL_SECONDS = 0.2
NO_PERSON_INFERENCE_INTERVAL_SECONDS = 1.0

app = FastAPI()
camera_in_use = False


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


def upload_telegram_video(filename, content_type, video_bytes):
    caption = "A person is no longer visible. Camera recording attached."
    for method, field, mime_type in (
        ("sendVideo", "video", content_type),
        ("sendDocument", "document", "application/octet-stream"),
    ):
        for attempt in range(3):
            try:
                response = requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}",
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                    files={field: (filename, video_bytes, mime_type)},
                    timeout=(10, 90),
                )
                response.raise_for_status()
                result = response.json()
                if result.get("ok"):
                    return {"ok": True, "method": method}
                print(f"Telegram {method} rejected {filename}: {result.get('description')}")
            except Exception as error:
                print(f"Telegram {method} attempt {attempt + 1} failed for {filename}: {error}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return {"ok": False}


def encode_frame(frame):
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not success:
        raise RuntimeError("Could not encode the camera frame")
    return base64.b64encode(encoded).decode("ascii")


def detect_and_encode(model, frame, state):
    now = time.monotonic()
    if now >= state["next_inference_time"]:
        try:
            result = model.predict(
                source=frame,
                classes=[0],
                imgsz=416,
                max_det=10,
                conf=0.35,
                verbose=False,
            )[0]
        except Exception as error:
            print(f"Detection failed; will retry: {error}")
            state["next_inference_time"] = now + NO_PERSON_INFERENCE_INTERVAL_SECONDS
            count = state["count"]
            image = frame
        else:
            count = len(result.boxes)
            state["count"] = count
            if count:
                state["last_detection_time"] = now
            state["next_inference_time"] = now + (
                PERSON_INFERENCE_INTERVAL_SECONDS
                if count
                else NO_PERSON_INFERENCE_INTERVAL_SECONDS
            )
            image = result.plot()
    else:
        count = state["count"]
        image = frame

    last_detection = state["last_detection_time"]
    recording = last_detection is not None and now - last_detection <= ABSENCE_GRACE_SECONDS
    return {"image": encode_frame(image), "count": count, "recording": recording}


@app.get("/", response_class=HTMLResponse)
def home():
        page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Human Detector</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; background: #101614; color: #eef4ef; }
body { max-width: 760px; margin: 0 auto; padding: 20px 16px 36px; }
h1 { margin: 4px 0 8px; font-size: 1.5rem; }
p, .muted { color: #b6c5bc; }
.view { width: 100%; aspect-ratio: 4 / 3; overflow: hidden; background: #25312c; border-radius: 8px; }
#display { display: block; width: 100%; height: 100%; object-fit: contain; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0; }
button { min-height: 46px; border: 0; border-radius: 6px; padding: 0 16px; font: inherit; font-weight: 650; }
button:disabled { opacity: .45; }
#start, .save { background: #baf36b; color: #172313; }
#stop, .quiet { background: #394740; color: #fff; }
#status { min-height: 1.5em; margin: 14px 0; color: #d8e6dc; }
#library { margin-top: 28px; border-top: 1px solid #394740; padding-top: 14px; }
.media-item { display: grid; grid-template-columns: minmax(90px, 150px) 1fr; gap: 12px; padding: 12px 0; border-bottom: 1px solid #394740; }
.preview { width: 100%; max-height: 120px; object-fit: contain; background: #202a25; }
.media-name { overflow-wrap: anywhere; font-weight: 600; }
.media-actions { display: flex; gap: 8px; margin-top: 8px; }
.media-actions button { min-height: 38px; padding: 0 12px; }
@media (max-width: 420px) { .media-item { grid-template-columns: 1fr; } .preview { max-height: 190px; } }
</style>
</head>
<body>
<h1>Human Detector</h1>
<p>Live person detection. Images and clips are stored in this browser.</p>
<div class="view"><canvas id="display" aria-label="Live detection view"></canvas></div>
<video id="camera" autoplay playsinline muted hidden></video>
<div class="controls">
    <button id="start">Start camera</button>
    <button id="stop" class="quiet" disabled>Stop</button>
    <button id="snapshot" class="quiet" disabled>Save image</button>
</div>
<div id="status" role="status">Camera is off.</div>
<div id="storage" class="muted">Browser storage status unavailable.</div>
<section id="library">
    <h2>Saved media</h2>
    <p class="muted">Stored in IndexedDB for this browser and site. Use Save to device to export a file.</p>
    <div id="media-list"></div>
</section>
<script>
const telegramEnabled = __TELEGRAM_ENABLED__;
const camera = document.querySelector('#camera');
const display = document.querySelector('#display');
const displayContext = display.getContext('2d');
const frameCanvas = document.createElement('canvas');
const frameContext = frameCanvas.getContext('2d');
const status = document.querySelector('#status');
const storageLabel = document.querySelector('#storage');
const startButton = document.querySelector('#start');
const stopButton = document.querySelector('#stop');
const snapshotButton = document.querySelector('#snapshot');
const mediaList = document.querySelector('#media-list');
const databasePromise = openDatabase();
let stream = null;
let socket = null;
let recorder = null;
let recorderStream = null;
let recorderChunks = [];
let recorderStartedAt = 0;
let serverRecording = false;
let restartRecorderAfterStop = false;
let objectUrls = [];

function openDatabase() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('human-detector-media', 1);
        request.onupgradeneeded = () => {
            const database = request.result;
            if (!database.objectStoreNames.contains('media')) {
                database.createObjectStore('media', { keyPath: 'id' });
            }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

function setStatus(message) {
    status.textContent = message;
}

function formatBytes(bytes) {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function updateStorageEstimate() {
    if (!navigator.storage || !navigator.storage.estimate) return;
    const estimate = await navigator.storage.estimate();
    const used = estimate.usage || 0;
    const quota = estimate.quota || 0;
    storageLabel.textContent = `Browser storage: ${formatBytes(used)} used${quota ? ` of ${formatBytes(quota)}` : ''}.`;
}

async function saveMedia(type, blob) {
    const extension = type === 'image' ? 'jpg' : (blob.type.includes('mp4') ? 'mp4' : 'webm');
    const date = new Date().toISOString().replace(/[:.]/g, '-');
    const record = {
        id: crypto.randomUUID(),
        type,
        name: `${type === 'image' ? 'human' : 'recording'}_${date}.${extension}`,
        createdAt: Date.now(),
        blob
    };
    const database = await databasePromise;
    await new Promise((resolve, reject) => {
        const transaction = database.transaction('media', 'readwrite');
        transaction.objectStore('media').put(record);
        transaction.oncomplete = resolve;
        transaction.onerror = () => reject(transaction.error);
        transaction.onabort = () => reject(transaction.error);
    });
    await refreshLibrary();
    await updateStorageEstimate();
    return record;
}

function readAllMedia() {
    return databasePromise.then(database => new Promise((resolve, reject) => {
        const request = database.transaction('media').objectStore('media').getAll();
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    }));
}

async function deleteMedia(id) {
    const database = await databasePromise;
    await new Promise((resolve, reject) => {
        const transaction = database.transaction('media', 'readwrite');
        transaction.objectStore('media').delete(id);
        transaction.oncomplete = resolve;
        transaction.onerror = () => reject(transaction.error);
    });
    await refreshLibrary();
    await updateStorageEstimate();
}

async function saveToDevice(record) {
    if (window.showSaveFilePicker) {
        try {
            const handle = await window.showSaveFilePicker({ suggestedName: record.name });
            const writable = await handle.createWritable();
            await writable.write(record.blob);
            await writable.close();
            return;
        } catch (error) {
            if (error.name === 'AbortError') return;
        }
    }
    const url = URL.createObjectURL(record.blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = record.name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
}

async function refreshLibrary() {
    const records = (await readAllMedia()).sort((left, right) => right.createdAt - left.createdAt);
    objectUrls.forEach(url => URL.revokeObjectURL(url));
    objectUrls = [];
    mediaList.replaceChildren();
    if (!records.length) {
        const empty = document.createElement('p');
        empty.className = 'muted';
        empty.textContent = 'No saved media yet.';
        mediaList.append(empty);
        return;
    }
    for (const record of records) {
        const url = URL.createObjectURL(record.blob);
        objectUrls.push(url);
        const item = document.createElement('article');
        item.className = 'media-item';
        const preview = document.createElement(record.type === 'image' ? 'img' : 'video');
        preview.className = 'preview';
        preview.src = url;
        preview.alt = record.name;
        if (record.type === 'video') {
            preview.controls = true;
            preview.playsInline = true;
            preview.preload = 'metadata';
        }
        const details = document.createElement('div');
        const name = document.createElement('div');
        name.className = 'media-name';
        name.textContent = record.name;
        const size = document.createElement('div');
        size.className = 'muted';
        size.textContent = `${formatBytes(record.blob.size)} · ${new Date(record.createdAt).toLocaleString()}`;
        const actions = document.createElement('div');
        actions.className = 'media-actions';
        const saveButton = document.createElement('button');
        saveButton.className = 'save';
        saveButton.textContent = 'Save to device';
        saveButton.addEventListener('click', () => saveToDevice(record));
        const removeButton = document.createElement('button');
        removeButton.className = 'quiet';
        removeButton.textContent = 'Delete';
        removeButton.addEventListener('click', () => deleteMedia(record.id));
        actions.append(saveButton, removeButton);
        details.append(name, size, actions);
        item.append(preview, details);
        mediaList.append(item);
    }
}

async function sendClipToTelegram(record) {
    if (!telegramEnabled) return;
    try {
        const response = await fetch('/telegram/upload', {
            method: 'POST',
            headers: {
                'Content-Type': record.blob.type || 'video/webm',
                'X-Filename': encodeURIComponent(record.name)
            },
            body: record.blob
        });
        if (!response.ok) throw new Error((await response.json()).detail || 'Telegram upload failed');
        setStatus('Recording saved in this browser and sent to Telegram.');
    } catch (error) {
        setStatus(`Recording saved locally; Telegram failed: ${error.message}`);
    }
}

function startRecorder() {
    if (recorder || !display.captureStream || !window.MediaRecorder) return;
    const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
        .find(type => MediaRecorder.isTypeSupported(type));
    recorderStream = display.captureStream(6);
    recorderChunks = [];
    recorder = new MediaRecorder(recorderStream, mimeType ? { mimeType } : undefined);
    recorderStartedAt = Date.now();
    const activeRecorder = recorder;
    activeRecorder.ondataavailable = event => {
        if (event.data && event.data.size) recorderChunks.push(event.data);
    };
    activeRecorder.onerror = event => setStatus(`Video recording error: ${event.error.message}`);
    activeRecorder.onstop = async () => {
        const blob = new Blob(recorderChunks, { type: activeRecorder.mimeType || 'video/webm' });
        recorderChunks = [];
        recorder = null;
        if (recorderStream) recorderStream.getTracks().forEach(track => track.stop());
        recorderStream = null;
        if (blob.size) {
            try {
                const record = await saveMedia('video', blob);
                await sendClipToTelegram(record);
                if (!telegramEnabled) setStatus(`Recording saved in this browser: ${record.name}`);
            } catch (error) {
                setStatus(`Could not store recording: ${error.message}`);
            }
        }
        if (restartRecorderAfterStop && serverRecording) startRecorder();
        restartRecorderAfterStop = false;
    };
    activeRecorder.start(1000);
}

function updateRecorder(recording) {
    serverRecording = recording;
    if (!recording) {
        restartRecorderAfterStop = false;
        if (recorder && recorder.state !== 'inactive') recorder.stop();
        return;
    }
    if (!recorder) {
        startRecorder();
    } else if (recorder.state === 'recording' && Date.now() - recorderStartedAt >= 30000) {
        restartRecorderAfterStop = true;
        recorder.stop();
    }
}

function captureFrame() {
    if (!socket || socket.readyState !== WebSocket.OPEN || camera.readyState < 2) return;
    const scale = Math.min(1, 640 / camera.videoWidth);
    frameCanvas.width = Math.round(camera.videoWidth * scale);
    frameCanvas.height = Math.round(camera.videoHeight * scale);
    frameContext.drawImage(camera, 0, 0, frameCanvas.width, frameCanvas.height);
    frameCanvas.toBlob(blob => {
        if (blob && socket && socket.readyState === WebSocket.OPEN) socket.send(blob);
    }, 'image/jpeg', 0.72);
}

function stopCamera() {
    serverRecording = false;
    if (recorder && recorder.state !== 'inactive') recorder.stop();
    const activeSocket = socket;
    socket = null;
    if (activeSocket && activeSocket.readyState < WebSocket.CLOSING) activeSocket.close();
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    camera.srcObject = null;
    startButton.disabled = false;
    stopButton.disabled = true;
    snapshotButton.disabled = true;
    setStatus('Camera is off.');
}

startButton.addEventListener('click', async () => {
    startButton.disabled = true;
    setStatus('Requesting camera access...');
    try {
        await databasePromise;
        if (navigator.storage && navigator.storage.persist) {
            try { await navigator.storage.persist(); } catch (error) { console.warn(error); }
        }
        stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                facingMode: { ideal: 'environment' },
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });
        camera.srcObject = stream;
        await camera.play();
        const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
        socket = new WebSocket(`${scheme}//${location.host}/stream`);
        socket.onopen = () => {
            stopButton.disabled = false;
            snapshotButton.disabled = false;
            setStatus('Loading model...');
        };
        socket.onmessage = event => {
            const message = JSON.parse(event.data);
            if (message.error) {
                stopCamera();
                setStatus(message.error);
            } else if (message.ready) {
                setStatus('Model ready. Detecting...');
                captureFrame();
            } else {
                const image = new Image();
                image.onload = () => {
                    display.width = image.naturalWidth;
                    display.height = image.naturalHeight;
                    displayContext.drawImage(image, 0, 0);
                    updateRecorder(message.recording);
                    setStatus(message.count
                        ? `Person detected: ${message.count}${message.recording ? ' · Recording' : ''}`
                        : `No person detected${message.recording ? ' · Recording' : ''}`);
                    captureFrame();
                };
                image.src = `data:image/jpeg;base64,${message.image}`;
            }
        };
        socket.onerror = () => setStatus('Camera connection failed.');
        socket.onclose = () => { if (stream) stopCamera(); };
    } catch (error) {
        stopCamera();
        setStatus(`Could not start camera: ${error.message}`);
    }
});

stopButton.addEventListener('click', stopCamera);
snapshotButton.addEventListener('click', () => {
    display.toBlob(async blob => {
        if (!blob) return;
        try {
            const record = await saveMedia('image', blob);
            setStatus(`Image saved in this browser: ${record.name}`);
        } catch (error) {
            setStatus(`Could not store image: ${error.message}`);
        }
    }, 'image/jpeg', 0.92);
});

refreshLibrary().then(updateStorageEstimate).catch(error => {
    setStatus(`Browser storage unavailable: ${error.message}`);
});
</script>
</body>
</html>"""
        return HTMLResponse(page.replace("__TELEGRAM_ENABLED__", json.dumps(bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID))))


@app.post("/telegram/upload")
async def telegram_upload(request: Request):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                raise HTTPException(status_code=503, detail="Telegram alerts are not configured")
        video_bytes = await request.body()
        if not video_bytes:
                raise HTTPException(status_code=400, detail="The recording is empty")
        filename = unquote(request.headers.get("x-filename", "recording.webm"))
        content_type = request.headers.get("content-type", "video/webm")
        result = await asyncio.to_thread(
                upload_telegram_video, filename, content_type, video_bytes
        )
        if not result["ok"]:
                raise HTTPException(status_code=502, detail="Telegram rejected the recording")
        return result


@app.websocket("/stream")
async def stream_camera(websocket: WebSocket):
    global camera_in_use
    if camera_in_use:
        await websocket.close(code=1013, reason="Camera is already in use")
        return

    camera_in_use = True
    state = {
        "next_inference_time": 0.0,
        "count": 0,
        "last_detection_time": None,
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
        camera_in_use = False