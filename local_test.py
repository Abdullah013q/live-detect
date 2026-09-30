import time
import cv2
import requests
from ultralytics import YOLO

# =======================================================
# 🚨 CONFIGURE YOUR TELEGRAM CREDENTIALS HERE 🚨
# =======================================================
TELEGRAM_BOT_TOKEN = "8709997524:AAGkz1J_o9hTLuj_sASWx1rWaF04TpKKcMo"
TELEGRAM_CHAT_ID = "-1003385021074"  # Must include the minus sign, e.g., "-10023456789"
ALERT_COOLDOWN_SECONDS = 15

# Track the last alert time to prevent spamming your chat group
last_alert_time = 0


def trigger_group_notification():
    global last_alert_time
    current_time = time.time()

    if current_time - last_alert_time > ALERT_COOLDOWN_SECONDS:
        last_alert_time = current_time
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": "⚠️ LOCAL SYSTEM ALERT: A person has been detected on the security camera feed!",
        }
        try:
            requests.post(url, json=payload, timeout=3)
            print("➡️ Telegram group alert sent successfully!")
        except Exception as e:
            print(f"❌ Failed to send Telegram alert: {e}")


def main():
    print("Initializing Tiny YOLOv8 Model...")
    model = YOLO("yolov8n.pt")

    # Start webcam stream (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    print("🚀 System Live! Press 'q' key in the video window to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame.")
            break

        # Run inference using the tiny model
        results = model(frame, stream=True)
        person_detected = False

        for r in results:
            # Draw boxes directly onto the live frame view
            frame = r.plot()

            # Inspect labels for person matches
            for box in r.boxes:
                if model.names[int(box.cls)] == "person":
                    person_detected = True

        if person_detected:
            trigger_group_notification()

        # Display the live window right on your Windows desktop
        cv2.imshow("Local AI Camera Test", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("System shut down cleanly.")


if __name__ == "__main__":
    main()
