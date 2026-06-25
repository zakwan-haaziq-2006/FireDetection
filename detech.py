import cv2 as cv
from ultralytics import YOLO
import time
import winsound

MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.3
DELAY_ALERT = 3
CLASS_NAMES = ['fire','other','smoke']
COLORS = {
    'fire': (0,0,255),
    'smoke': (128,128,128),
    'other': (0,255,255)
}



def get_severity(x1,y1,x2,y2,frame_area):
    width = x2 - x1
    height = y2 - y1
    box_area = width * height
    
    ratio = box_area / frame_area
    
    if ratio < 0.02 :
        return "LOW",(0,255,0)
    elif ratio < 0.08:
        return "MEDIUM",(0,165,255) 
    else:
        return "CRITICAL",(0,0,255)
    


area_history = []    
def get_spread_prediction(curr_area):
    area_history.append(curr_area)
    
    if len(area_history) > 30 :
        area_history.pop(0)
    
    if len(area_history) >= 5 :
        growth = (area_history[-1] - area_history[0])/len(area_history)
        pred_area = curr_area + (growth * 150)
        return max(pred_area,curr_area)
    return curr_area


first_start_time = None
already_triggered = False
def check_alert(fire_detected):
    global first_start_time,already_triggered
    
    if fire_detected:
        if first_start_time is None:
            first_start_time = time.time()
            
        elapsed = time.time() - first_start_time
              
        if elapsed >= DELAY_ALERT and not already_triggered:
            already_triggered = True
            for _ in range(3):
                winsound.Beep(1000,300)
                time.sleep(0.1)
            return True
    if not fire_detected:
        first_start_time = None
        already_triggered = False
    return False


def draw_overley(frame,label,conf,x1,y1,x2,y2,severity,sev_color):
    color = COLORS[label.lower()]
    cv.rectangle(frame,(x1,y1),(x2,y2),color,1)
    text = f"Label : {label} : {conf:.0%} | severity : {severity}"
    (tw,th),_ = cv.getTextSize(text,cv.FONT_HERSHEY_SIMPLEX,0.6,2)
    cv.rectangle(frame,(x1,y1-th-8),(x1+tw,y1),color,-1)
    cv.putText(frame,text,(x1,y1-5),cv.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    cv.putText(frame,f"SEVERITY : {severity}",(frame.shape[1]-250,40),cv.FONT_HERSHEY_SIMPLEX,0.8,sev_color,2)
    
    

def draw_spread(frame,x1,y1,x2,y2,pred_area):
    cx = (x1 + x2)//2
    cy = (y1+ y2)//2
    
    side = pred_area**0.5
    half = int(side)//2
    
    px1 = max(0,cx - half)
    py1 = max(0,cy - half)
    px2 = min(frame.shape[1],cx + half)
    py2 = min(frame.shape[0],cy + half)
    
    for i in range(int(px1),int(px2),10):
        cv.line(frame,(i,int(py1)),(i+5,int(py1)),(0,165,255),1)
        cv.line(frame,(i,int(py2)),(i+5,int(py2)),(0,165,255),1)
        
    for i in range(int(py1),int(py2),10):
        cv.line(frame,(int(px1),i),(int(px1),i+5),(0,165,255),1)
        cv.line(frame,(int(px2),i),(int(px2),i+5),(0,165,255),1)
        
    cv.putText(frame,"Predicted spread (5s)",(int(px1),int(py1)-5),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,165,255),1)
    
    
def main():
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded")
    
    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH,1280)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT,720)
    print("✅ Camera opened")
    
    while True:
        ret,frame = cam.read()
        if not ret:
            print("❌ Camera error")
            break
        
        frame_area = frame.shape[0] * frame.shape[1]
        fire_detected = False
        
        results = model(frame,conf=CONFIDENCE_THRESHOLD,verbose=False)
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                label = CLASS_NAMES[cls]
                
                # LOG every detection
                print(f"🔍 Detected: {label} | conf: {conf:.0%} | box: ({x1},{y1},{x2},{y2})")
                
                sev,sev_color = get_severity(x1,y1,x2,y2,frame_area)
                print(f"⚠️  Severity: {sev}")
                
                if label == 'fire':
                    fire_detected = True
                    pred_area = get_spread_prediction((x2-x1)*(y2-y1))
                    print(f"🔥 Fire detected | pred_area: {int(pred_area)}px")
                    draw_spread(frame,x1,y1,x2,y2,pred_area)
                    
                draw_overley(frame,label,conf,x1,y1,x2,y2,sev,sev_color)
                
        check_alert(fire_detected)
                
        if already_triggered:
            print("🚨 ALERT TRIGGERED")
            cv.rectangle(frame,(0,0),(frame.shape[1]-1,frame.shape[0]-1),(0,0,255),8)
            cv.putText(frame,"FIRE ALERT!!",(frame.shape[1]//2 - 150,frame.shape[0]//2),cv.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)
            
        cv.imshow('Fire Detection',frame)
        
        if cv.waitKey(1) & 0xff == ord('q'):
            break
    
    cam.release()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()
    
    