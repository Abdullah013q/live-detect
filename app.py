import io
import requests
import streamlit as st

# =======================================================
# 🚨 CONFIGURE YOUR TELEGRAM CREDENTIALS HERE 🚨
# =======================================================
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = (
    "YOUR_GROUP_ID_HERE"  # Must include minus sign, e.g., "-10023456789"
)

# Set page configuration
st.set_page_config(
    page_title="Click-to-Send Security Cam", page_icon="📸", layout="centered"
)

st.title("📸 Quick Capture Security Camera")
st.write(
    "Click the button inside the camera box to snap a photo. It will be sent directly to your Telegram group chat."
)


# Function to send image binary data to Telegram API
def send_photo_to_telegram(image_bytes):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "caption": "📸 Manual Alert: A snapshot has been captured from the live interface widget!"}

    # Pass the binary image buffer layout directly into the network payload
    files = {"photo": ("snapshot.jpg", image_bytes, "image/jpeg")}

    try:
        response = requests.post(url, data=payload, files=files, timeout=10)
        if response.status_code == 200:
            st.success("✅ Image sent successfully to Telegram Group!")
        else:
            st.error(
                f"❌ Telegram API Error: {response.text} (Code: {response.status_code})"
            )
    except Exception as e:
        st.error(f"❌ Network request failed: {e}")


# Streamlit native camera capture widget layout
# This securely prompts user browser permissions without breaking ports
picture = st.camera_input("Camera Grid View")

if picture is not None:
    # Read the image file buffer stream from the layout
    bytes_data = picture.getvalue()

    # Visual separation helper container
    st.image(picture, caption="Captured Image Preview", use_container_width=True)

    # Action submission switch
    if st.button("📤 Send Captured Image to Telegram", type="primary"):
        with st.spinner("Uploading photo to group stream..."):
            send_photo_to_telegram(bytes_data)
