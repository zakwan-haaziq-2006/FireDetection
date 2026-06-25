import requests
import cv2 as cv
import streamlit as st
import platform
from dotenv import load_dotenv
load_dotenv()
import os
API_URL = os.getenv("API_URL")# replace with your Render URL

# Cross-platform beep
if platform.system() == "Windows":
    import winsound
    def beep():
        for _ in range(3):
            winsound.Beep(1500, 300)
else:
    def beep():
        pass

# UI
st.title("🔥 FIRE DETECTION SYSTEM")
st.header("Real-time Fire & Smoke Detection")
st.caption("Powered by YOLOv8 | 99% mAP50")

# Session state
if "running" not in st.session_state:
    st.session_state.running = False
if "alert_triggered" not in st.session_state:
    st.session_state.alert_triggered = False
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "detections" not in st.session_state:
    st.session_state.detections = []

# Buttons
col1, col2 = st.columns(2)
if col1.button("▶ Start Detection"):
    st.session_state.running = True
    st.session_state.alert_triggered = False
if col2.button("⏹ Stop Detection"):
    st.session_state.running = False

# Placeholders
frame_placeholder = st.empty()
alert_placeholder = st.empty()
stats_placeholder = st.empty()

if st.session_state.running:
    cap = cv.VideoCapture(0)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        st.session_state.frame_count += 1
        fire_detected = False

        # Send every 5th frame to API
        if st.session_state.frame_count % 5 == 0:
            try:
                _, buffer = cv.imencode('.jpg', frame)
                bytes_frame = buffer.tobytes()

                response = requests.post(
                    API_URL + "/detect",
                    files={'file': ("frame.jpg", bytes_frame, "image/jpeg")},
                    timeout=5
                )
                data = response.json()
                st.session_state.detections = data['detections']

            except Exception as e:
                print(f"API error: {e}")

        # Draw detections on frame
        for det in st.session_state.detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            conf = det['confidence']
            severity = det['severity']

            if label == 'fire':
                fire_detected = True

            # color by severity
            if severity == "CRITICAL":
                color = (0, 0, 255)
            elif severity == "MEDIUM":
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            cv.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            # label background
            text = f"{label} {conf:.0%} | {severity}"
            (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv.rectangle(frame, (x1, y1-th-8), (x1+tw, y1), color, -1)
            cv.putText(frame, text, (x1, y1-5),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Alert logic
        if fire_detected and not st.session_state.alert_triggered:
            st.session_state.alert_triggered = True
            beep()
            alert_placeholder.error("🚨 FIRE ALERT TRIGGERED! Evacuate immediately!")

        if not fire_detected:
            st.session_state.alert_triggered = False
            alert_placeholder.empty()

        # Alert border on frame
        if st.session_state.alert_triggered:
            cv.rectangle(frame, (0,0),
                        (frame.shape[1]-1, frame.shape[0]-1),
                        (0,0,255), 8)
            cv.putText(frame, "FIRE ALERT!!",
                      (frame.shape[1]//2-150, frame.shape[0]//2),
                      cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

        # Stats sidebar
        stats_placeholder.info(
            f"Frames processed: {st.session_state.frame_count} | "
            f"Detections: {len(st.session_state.detections)}"
        )

        # Display frame
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, use_container_width=True)

    cap.release()