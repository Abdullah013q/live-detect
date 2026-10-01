import time
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"

from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ==========================================
# 🛑 ENTER YOUR TELEGRAM CREDENTIALS HERE 🛑
# ==========================================
TELEGRAM_BOT_TOKEN = "8709997524:AAGkz1J_o9hTLuj_sASWx1rWaF04TpKKcMo"
TELEGRAM_CHAT_ID = "-1003385021074"
ALERT_COOLDOWN_SECONDS = (
    15  # Minimum time to wait before sending another alert
)

# Initialize a simple session timestamp to prevent spamming alerts
if "last_alert_time" not in st.session_state:
    st.session_state["last_alert_time"] = 0

# Page Configuration & UI Layout
st.set_page_config(
    page_title="Security Camera AI Alert", page_icon="🚨", layout="centered"
)

st.title("🚨 Live Security Detection with Telegram Alerts")
st.write(
    "This app tracks the video feed and will instantly message your Telegram app if a **Person** enters the frame."
)


# Cached Model Loading
@st.cache_resource
def load_tiny_model():
    return YOLO("yolov8n.pt")


model = load_tiny_model()


# Helper function to send the Telegram Notification
def send_telegram_alert():
    current_time = time.time()
    # Check if enough time has passed since the last alert to avoid spamming
    if (
        current_time - st.session_state["last_alert_time"]
        > ALERT_COOLDOWN_SECONDS
    ):
        st.session_state["last_alert_time"] = current_time
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": "⚠️ WARNING: A person has been detected on your live camera feed!",
        }
        try:
            # Send the request in the background
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            pass  # Fail silently to prevent the video stream from crashing


# WebRTC Frame Processing Callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run the tiny model on the frame
    results = model(img, stream=True)

    person_detected = False
    annotated_img = img

    for r in results:
        annotated_img = r.plot()

        # Check all detected objects in the frame
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]

            # In the COCO dataset, index 0 corresponds to 'person'
            if label == "person":
                person_detected = True

    # If a person is found, trigger the alert function
    if person_detected:
        send_telegram_alert()

    return frame.from_ndarray(annotated_img, format="bgr24")


# 5. Build WebRTC Secure Interface
webrtc_streamer(
    key="yolov8-live-group-security",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
            {"urls": "stun:stun2.l.google.com:19302"},
            {"urls": "stun:stun3.l.google.com:19302"},
            {"urls": "stun:stun4.l.google.com:19302"},
        ]
    },  # Cleaned public network routing format
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,  # Decouples video UI elements from computation
)


st.caption("Click **Start** to run the security alert system.")
