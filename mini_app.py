import hashlib
import os
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, ImageOps

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


MODEL_PATH = Path(__file__).with_name("yolov8n.pt")
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_SIDE = 960
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

st.set_page_config(page_title="Human Detector", page_icon="📷", layout="centered")
st.markdown(
    """
    <style>
    .block-container { max-width: 680px; padding: 1rem 1rem 2rem; }
    [data-testid="stCameraInput"] button,
    [data-testid="stFileUploader"] button,
    [data-testid="stButton"] button { min-height: 48px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the detection model...")
def load_model():
    if YOLO is None:
        raise RuntimeError("Ultralytics is missing. Install it with: pip install ultralytics")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


def prepare_image(image_bytes):
    image = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return image


def send_telegram_photo(image, count):
    image_file = BytesIO()
    image.save(image_file, format="JPEG", quality=82, optimize=True)
    image_file.seek(0)
    response = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": f"Human detected. Count: {count}."},
        files={"photo": ("human-detection.jpg", image_file, "image/jpeg")},
        timeout=(5, 30),
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("ok"):
        raise RuntimeError(result.get("description", "Telegram rejected the image"))


st.title("Human Detector")
st.caption("Single-photo detection · optimized for phone cameras")

camera_photo = st.camera_input("Take a photo")
uploaded_photo = st.file_uploader(
    "Or choose a photo",
    type=("jpg", "jpeg", "png"),
    accept_multiple_files=False,
)
photo = camera_photo or uploaded_photo

if photo is not None:
    image_bytes = photo.getvalue()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        st.error("This image is larger than 8 MB. Choose a smaller image and try again.")
    else:
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        image = prepare_image(image_bytes)
        st.image(image, caption="Ready for analysis", use_container_width=True)
        send_alert = st.checkbox(
            "Send a detected-human photo to Telegram",
            value=bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            disabled=not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        )

        if st.button("Analyze photo", type="primary", use_container_width=True):
            try:
                model = load_model()
                with st.spinner("Checking for a human..."):
                    result = model.predict(
                        source=image,
                        classes=[0],
                        imgsz=320,
                        max_det=10,
                        conf=0.35,
                        verbose=False,
                    )[0]
                count = len(result.boxes)
                annotated = Image.fromarray(result.plot()[..., ::-1])
                st.session_state["analysis"] = {
                    "digest": image_digest,
                    "count": count,
                    "image": annotated,
                }
                st.session_state["send_alert"] = send_alert
            except Exception as error:
                st.error(f"Could not analyze this photo: {error}")

        analysis = st.session_state.get("analysis")
        if analysis and analysis["digest"] == image_digest:
            st.image(analysis["image"], caption="Detection result", use_container_width=True)
            if analysis["count"]:
                st.success(f"Human detected: {analysis['count']}")
                if st.session_state.get("send_alert"):
                    alert_key = f"telegram_sent_{image_digest}"
                    if not st.session_state.get(alert_key):
                        try:
                            send_telegram_photo(analysis["image"], analysis["count"])
                            st.session_state[alert_key] = True
                            st.success("Detection photo sent to Telegram.")
                        except Exception as error:
                            st.error(f"Telegram delivery failed: {error}")
            else:
                st.info("No human detected.")
else:
    st.info("Capture a photo or choose one from your device to begin.")

st.caption("Inference runs on the Streamlit host. This mobile version analyzes still photos, not live video.")
