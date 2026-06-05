import io
import os
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import base64
from ultralytics import YOLO

app = FastAPI(
    title="Kidney Stone Detection API (YOLO26m)",
    description="Backend API matching the exact preprocessing and inference sequence of the training notebook."
)

# ----------------------------------------------------
# 📂 1. MODEL INITIALIZATION & PATH CONFIGURATION
# ----------------------------------------------------
# Point directly to your local best.pt file inside your workspace folder
MODEL_PATH = Path("best.pt") 

if not MODEL_PATH.exists():
    # Fallback to absolute path configuration if not in current root
    MODEL_PATH = Path(r"D:\development\python\Kidney stone detection\weights\best.pt")

print(f"⏳ Loading YOLO26m model weights from: {MODEL_PATH}")
try:
    model = YOLO(str(MODEL_PATH))
    print("✅ Model loaded successfully onto global memory!")
except Exception as e:
    print(f"❌ Failed to load model weights: {e}")
    model = None

# Global constant parameters matching your notebook setup
IMG_SIZE_YOLO = 640

# Pydantic schema for response structure validation
class DetectionResult(BaseModel):
    class_id: int
    bbox: List[float]
    score: float

@app.post("/predict")
async def predict_kidney_stone(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO Model is uninitialized or weight file path is missing.")
        
    # Read raw uploaded file stream bytes
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid file stream format.")
        
    # Convert BGR (OpenCV default) to RGB to match notebook pipeline initial state
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ----------------------------------------------------
    # 🔹 NOTEBOOK SEQUENCE: Image Enhancement Steps
    # ----------------------------------------------------
    # Step 1: Convert to grayscale for CLAHE
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 2: Median Filtering (Noise Removal)
    median = cv2.medianBlur(gray, 5)

    # Step 3: CLAHE (Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(median)

    # Step 4: Convert back to RGB (3 channels)
    processed_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    # ----------------------------------------------------
    # 🚀 NOTEBOOK SEQUENCE: YOLO Inference Execution
    # ----------------------------------------------------
    # Convert back to standard OpenCV BGR array right before passing to Ultralytics pipeline
    inference_img = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
    
    results = model.predict(
        source=inference_img, 
        imgsz=IMG_SIZE_YOLO, 
        conf=0.25, 
        verbose=False
    )
    
    res = results[0]
    
    # ----------------------------------------------------
    # 📊 NOTEBOOK SEQUENCE: Extract Coordinates & Plot Output
    # ----------------------------------------------------
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
    scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
    labels = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else []
    
    preds_json = []
    for box, score, cls in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        preds_json.append({
            "class_id": int(cls),
            "bbox": [x1, y1, x2, y2],
            "score": float(score)
        })
        
    # Generate the plotted diagnostic validation image array with bounding boxes natively
    plotted_bgr = res.plot()
    
    # Encode output array matrix to PNG stream bytes
    _, img_encoded = cv2.imencode('.png', plotted_bgr)
    base64_image_str = base64.b64encode(img_encoded).decode('utf-8')
    
    # Return composite standard response package payload
    return JSONResponse(content={
        "status": "success",
        "detections": preds_json,
        "annotated_img": base64_image_str
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)