from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2 as cv
from dotenv import load_dotenv
import os


load_dotenv()

CLASS_NAMES = ['fire', 'other', 'smoke']

def get_severity(x1, y1, x2, y2, frame_area):
    box_area = (x2-x1) * (y2-y1)
    ratio = box_area / frame_area
    if ratio < 0.02:
        return "LOW", (0,255,0)
    elif ratio < 0.08:
        return "MEDIUM", (0,165,255)
    else:
        return "CRITICAL", (0,0,255)



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)



try:
    print("Loading Model......")
    model = YOLO(os.getenv("MODEL_PATH","best.pt"))
    print("✅Model Loaded Successfully")
except Exception as e:
    print("❌MODEL FAILED TO LOAD")
    
@app.get("/")
def root():
    return {"status":"Running"}

@app.get('/health')
def health_chk():
    return {"status":"Healthy",}
    
    



@app.post('/detect')
async def detect(file : UploadFile = File(...)):
    contents = await file.read()
    print(f"Recieved {file.filename} | size : {len(contents)} bytes")
    arr = np.frombuffer(contents,np.uint8)
    img = cv.imdecode(arr,cv.IMREAD_COLOR)
    detections = []
    results = model(img,conf=float(os.getenv("CONFIDENCE_THRESHOLD","0.15")),verbose=False)
    img_area = img.shape[0] * img.shape[1]
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            conf = round(float(box.conf),2)
            (x1,y1,x2,y2) = map(int,box.xyxy[0])
            label = CLASS_NAMES[cls]
    
            servity,_ = get_severity(x1,y1,x2,y2,img_area)
            detections.append({'label':label,'confidence':conf,'box':(x1,y1,x2,y2),'severity':servity})
    
    return {'detections': detections}
            