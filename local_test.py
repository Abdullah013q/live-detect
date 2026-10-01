import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import requests
from ultralytics import YOLO

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ABSENCE_GRACE_SECONDS = 2
NO_HUMAN_INFERENCE_INTERVAL_SECONDS = 1.0
IDLE_LOOP_SLEEP_SECONDS = 0.05
RECORDING_FPS = 12.0
RECORDING_MAX_WIDTH = 640
RECORDING_SEGMENT_SECONDS = 30
RECORDINGS_DIR = Path("recordings")


def upload_telegram_file(method, field, video_path, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    content_type = "video/mp4" if method == "sendVideo" else "application/octet-stream"

    for attempt in range(3):
        try:
            with video_path.open("rb") as video_file:
                response = requests.post(
                    url,
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                    files={field: (video_path.name, video_file, content_type)},
                    timeout=(10, 90),
                )
            response.raise_for_status()
            result = response.json()
            if result.get("ok"):
                return True
            print(f"Telegram {method} rejected {video_path.name}: {result.get('description')}")
        except Exception as error:
            print(f"Telegram {method} attempt {attempt + 1} failed for {video_path.name}: {error}")

        if attempt < 2:
            time.sleep(2 ** attempt)

    return False


def send_disappearance_alert(video_paths):
    caption = "A person is no longer visible. Camera recording attached."
    for video_path in video_paths:
        if upload_telegram_file("sendVideo", "video", video_path, caption):
            print(f"Telegram video sent: {video_path}")
            continue
        if upload_telegram_file("sendDocument", "document", video_path, caption):
            print(f"Telegram recording sent as a document: {video_path}")
            continue

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
            print(f"Telegram could not accept the clip; local copy retained at {video_path}")
        except Exception as error:
            print(f"Failed to notify Telegram about {video_path}: {error}")


def main():
    if not TELEGRAM_BOT_TOKEN:
        print("Set the TELEGRAM_BOT_TOKEN environment variable before starting.")
        return

    print("Initializing Tiny YOLOv8 Model...")
    model = YOLO("yolov8n.pt")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    cap = None
    video_writer = None
    recording_path = None
    recording_paths = []
    segment_started_at = None
    next_record_frame_time = 0.0
    last_detection_time = None
    human_present = False
    next_inference_time = 0.0
    failed_reads = 0
    telegram_executor = ThreadPoolExecutor(max_workers=1)

    print("🚀 System Live! Press 'q' key in the video window to stop.")

    try:
        while True:
            if cap is None or not cap.isOpened():
                print("Camera unavailable; retrying in 3 seconds.")
                if cap is not None:
                    cap.release()
                time.sleep(3)
                cap = cv2.VideoCapture(0)
                continue

            ret, frame = cap.read()
            if not ret:
                failed_reads += 1
                if failed_reads >= 5:
                    print("Camera read failed; reconnecting.")
                    cap.release()
                    cap = None
                    failed_reads = 0
                else:
                    time.sleep(IDLE_LOOP_SLEEP_SECONDS)
                continue
            failed_reads = 0

            current_time = time.monotonic()
            if current_time >= next_inference_time:
                try:
                    results = model(frame, stream=True, classes=[0], imgsz=416, verbose=False)
                    human_detected = False
                    for result in results:
                        frame = result.plot()
                        if any(model.names[int(box.cls)] == "person" for box in result.boxes):
                            human_detected = True
                except Exception as error:
                    print(f"Detection failed; will retry: {error}")
                    next_inference_time = current_time + NO_HUMAN_INFERENCE_INTERVAL_SECONDS
                else:
                    human_present = human_detected
                    if human_present:
                        last_detection_time = current_time
                    next_inference_time = current_time + (
                        0 if human_present else NO_HUMAN_INFERENCE_INTERVAL_SECONDS
                    )

            if human_present and video_writer is None:
                recording_path = RECORDINGS_DIR / (
                    f"human_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
                )
                recording_frame = frame
                height, width = recording_frame.shape[:2]
                if width > RECORDING_MAX_WIDTH:
                    height = int(height * RECORDING_MAX_WIDTH / width)
                    recording_frame = cv2.resize(
                        recording_frame, (RECORDING_MAX_WIDTH, height)
                    )
                video_writer = cv2.VideoWriter(
                    str(recording_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    RECORDING_FPS,
                    (recording_frame.shape[1], recording_frame.shape[0]),
                )
                if video_writer.isOpened():
                    segment_started_at = current_time
                    next_record_frame_time = current_time
                else:
                    print("Could not start MP4 recording.")
                    video_writer.release()
                    video_writer = None
                    recording_path = None

            if video_writer is not None:
                if not human_present and current_time - last_detection_time > ABSENCE_GRACE_SECONDS:
                    video_writer.release()
                    video_writer = None
                    if recording_path.exists() and recording_path.stat().st_size:
                        recording_paths.append(recording_path)
                    recording_path = None
                    if recording_paths:
                        telegram_executor.submit(send_disappearance_alert, tuple(recording_paths))
                        recording_paths = []
                else:
                    recording_frame = frame
                    height, width = recording_frame.shape[:2]
                    if width > RECORDING_MAX_WIDTH:
                        height = int(height * RECORDING_MAX_WIDTH / width)
                        recording_frame = cv2.resize(
                            recording_frame, (RECORDING_MAX_WIDTH, height)
                        )
                    if current_time >= next_record_frame_time:
                        video_writer.write(recording_frame)
                        next_record_frame_time = current_time + 1 / RECORDING_FPS

                    if current_time - segment_started_at >= RECORDING_SEGMENT_SECONDS:
                        video_writer.release()
                        video_writer = None
                        if recording_path.exists() and recording_path.stat().st_size:
                            recording_paths.append(recording_path)
                        recording_path = None

            cv2.imshow("Local AI Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if not human_present:
                time.sleep(IDLE_LOOP_SLEEP_SECONDS)
    finally:
        if video_writer is not None:
            video_writer.release()
            if recording_path.exists() and recording_path.stat().st_size:
                recording_paths.append(recording_path)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        telegram_executor.shutdown(wait=False)

    print("System shut down cleanly.")


if __name__ == "__main__":
    main()
