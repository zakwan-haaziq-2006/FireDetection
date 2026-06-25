
import requests
import cv2 as cv
import streamlit as st
import winsound

API_URL = "http://localhost:8000"

st.title("🔥 FIRE DETECTION SYSTEM")
st.header("REAL TIME FIRE & SMOKE DETECTION")

if "running" not in st.session_state:
    st.session_state.running = False
if "alert_triggered" not in st.session_state:
    st.session_state.alert_triggered = False
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

col1, col2 = st.columns(2)
if col1.button("▶ Start"):
    st.session_state.running = True
    st.session_state.alert_triggered = False
if col2.button("⏹ Stop"):
    st.session_state.running = False

frame_placeholder = st.empty()
alert_placeholder = st.empty()

if st.session_state.running:
    cap = cv.VideoCapture(0)
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        st.session_state.frame_count += 1
        fire_detected = False

        if st.session_state.frame_count % 5 == 0:
            _, buffer = cv.imencode('.jpg', frame)
            bytes_frame = buffer.tobytes()

            try:
                response = requests.post(
                    API_URL + "/detect",
                    files={'file': ("frame.jpg", bytes_frame, "image/jpeg")}
                )
                data = response.json()
                detections = data['detections']

                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    label = det['label']
                    conf = det['confidence']
                    severity = det['severity']

                    if label == 'fire':
                        fire_detected = True

                    # box color by severity
                    if severity == "CRITICAL":
                        color = (0, 0, 255)
                    elif severity == "MEDIUM":
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)

                    cv.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv.putText(frame, f"{label} {conf:.0%} | {severity}",
                               (x1, y1-5),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # alert
                if fire_detected and not st.session_state.alert_triggered:
                    st.session_state.alert_triggered = True
                    for _ in range(3):
                        winsound.Beep(1500, 300)
                    alert_placeholder.error("🚨 FIRE ALERT TRIGGERED!")

                if not fire_detected:
                    st.session_state.alert_triggered = False
                    alert_placeholder.empty()

            except Exception as e:
                print(f"API error: {e}")

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, use_container_width=True)

    cap.release()