import requests
import cv2 as cv
import streamlit as st

API_URL = "http://localhost:8000"

st.title("FIRE DETECTION SYSTEM")
st.header("REAL TIME FIRE & SMOKE DETECTION")

# Session state to control loop
if "running" not in st.session_state:
    st.session_state.running = False

# Buttons OUTSIDE loop
col1, col2 = st.columns(2)
if col1.button("▶ Start"):
    st.session_state.running = True
if col2.button("⏹ Stop"):
    st.session_state.running = False

frame_placeholder = st.empty()

if st.session_state.running:
    cap = cv.VideoCapture(0)
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break
        
        
        
        _,buffer = cv.imencode('.jpg',frame)
        bytes_frame = buffer.tobytes()
        
        response = requests.post(API_URL+"/detect",
                                 files={'file':("frame.jpg",bytes_frame,'image/jpeg')}
                                 )
        data = response.json()
        detections = data['detections']
        
        for det in detections:
            x1,y1,x2,y2 = det['box']
            label = det['label']
            conf = det['confidence']
            seveirity = det['severity']
            
            cv.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv.putText(frame,f"Label {label} : {conf:.0%} | severity : {seveirity}",(x1,y1-5),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,165,255),2)
    
        frame_rgb= cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, use_container_width=True)        
    cap.release()