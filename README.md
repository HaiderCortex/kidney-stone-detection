# 🩺 Kidney Stone Object Detection using YOLO26m

An end-to-end deep learning and computer vision pipeline engineered to detect and locate kidney stones from medical CT scan slices. This repository contains the source code for a high-performance **YOLO26m** object detection model, backed by a robust **FastAPI** application layout and an interactive **Streamlit** diagnostic dashboard.

---

## **🚀 Key Features**
* **Custom Medical Preprocessing:** Built-in noise reduction and local contrast optimization matching the exact training distribution.
* **State-of-the-Art Detection:** Integrated with a fine-tuned YOLO26m architecture for tight bounding-box localization.
* **Production-Ready API:** High-throughput FastAPI backend handling parallel streaming inference requests.
* **Interactive Dashboard:** Medical control interface designed in Streamlit for clear side-by-side diagnostic visualization.

------------------------------------

## **📂 Repository Structure**
```text
├── app.py              # FastAPI backend script containing image preprocessing & inference loop
├── frontend.py         # Streamlit dashboard interface displaying interactive visual results
├── best.pt             # Fine-tuned YOLO26m model weights (place in root folder)
├── requirements.txt    # Application dependency specifications
└── README.md           # Project documentation
```
-------------------------------------

## **⚙️ Core Pipeline Architecture**
The application executes a deterministic medical computer vision sequence on every uploaded image to prevent distribution shift and maintain diagnostic consistency:
[User Scan Upload] ──> (Streamlit UI) ──> [HTTP POST API Request]
                                                  │
                                                  ▼
                                          (FastAPI Backend)
                                                  │
                                      ┌───────────┴───────────┐
                                      │  1. Gray Conversion   │
                                      │  2. Median Filter     │
                                      │  3. CLAHE Contrast    │
                                      │  4. RGB Conversion    │
                                      └───────────┬───────────┘
                                                  │
                                                  ▼
                                         [YOLO26m Inference]
                                                  │
                                                  ▼
[Render Layout & Metrics] <── (Base64 + JSON) <── [Extract BBoxes & Plot]

------------------------------

## **🔹 Notebook Preprocessing Sequence**
1.Grayscale Conversion: Isolates raw structural density maps.
2.Median Blur Filtering (kernel=5): Suppresses salt-and-pepper noise while preserving critical edge boundaries.
3.CLAHE (clipLimit=2.0, tileGridSize=(8,8)): Enhances local contrast to draw out subtle calcification features.
4.RGB Reconstruction: Converts the optimized matrices back to standard channels for YOLO processing.

-------------------------------

## **🛠️ Installation & Setup**
### **1. Clone the Workspace**
  git clone [https://github.com/your-username/kidney-stone-detection.git](https://github.com/your-username/kidney-stone-       detection.git)
  cd kidney-stone-detection

### **2. Configure Dependencies**
Ensure you have Python 3.9+ installed, then run:
     pip install -r requirements.txt
Dependencies include: ultralytics, fastapi, streamlit, opencv-python, numpy, requests, pydantic, and uvicorn.

### **3. Add Model Weights**
Ensure your fine-tuned best.pt file is saved directly into the root directory of this workspace:
    D:\development\python\Kidney stone detection\weights\best.pt  --> copy to root workspace
---------------------------------
## **🛫 Running the Application**
To run the full stack, open two separate terminal instances in your workspace environment:

### **Step 1: Initialize the FastAPI Server**
     python app.py
The backend server will host locally at http://127.0.0.1:8000. You can view the interactive interactive API documentation at http://127.0.0.1:8000/docs.

### **Step 2: Launch the Streamlit Dashboard**
         streamlit run frontend.py
The interface will automatically deploy in your default web browser at http://localhost:8501.

-------------------------------

## **📊 Evaluation & Metrics**
The underlying network was fine-tuned on a kidney stone CT dataset with bounding box annotations over 50 epochs at an image resolution of 640x640 pixels, utilizing a batch size of 16.

The validation loop records localized structural accuracy scores across:
mAP@50 (Bounding Box Accuracy)
Precision & Recall Profiles
Inference Speed Latency (FPS)

--------------------------------

## **📜 License**
This project is intended for educational, research, and engineering deployment purposes. All clinical validations should be confirmed by certified radiologists.
