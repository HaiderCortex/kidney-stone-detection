# Automated Kidney Stone Detection & Segmentation
**AI-Powered Diagnostics & Image Segmentation (FYP 2026)**

An advanced, end-to-end artificial intelligence healthcare platform designed to enhance diagnostic accuracy and expedite clinical workflows. This project utilizes a dual-stage neural network pipeline combining **YOLO** for real-time object localization and **UNet++** for precise pixel-level medical image segmentation.

## 👥 Team Members (Group 31)
* **Furqan Haider** (Group Leader)
* **Mashal Hassan**
* **Muhammad Shaheer**
* **Supervisor:** Dr. Sanaullah Khan (Institute of Computing, KUST)

---

## 🏗️ Overall System Architecture & Pipeline
The workflow utilizes a cascaded approach: medical CT scans undergo a custom 4-stage preprocessing bridge to eliminate noise. The YOLO network identifies Regions of Interest (ROIs), which are then dynamically cropped and fed into the UNet++ model for fine-grained segmentation.

```mermaid
graph TD
    A[Raw CT Scan] --> B(Stage 1: Grayscale Conversion)
    B --> C(Stage 2: Median Blur 5x5 Kernel)
    C --> D(Stage 3: CLAHE Contrast Optimization)
    D --> E(Stage 4: RGB Restoration)
    E --> F{YOLO Detection Layer}
    F -->|Draws Bounding Box| G[ROI Cropping]
    G --> H{UNet++ Segmentation Layer}
    H -->|Pixel-Level Mapping| I[Final Diagnostic Image Overlay]

## 🎯 Phase 1: Object Detection (YOLO)

The detection module is responsible for the rapid, real-time localization of renal calculi within clinical CT slices.

### Approach & Pipeline

* **Model:** YOLOv26 (Ultralytics ecosystem).
* **Image Preprocessing:** Grayscale conversion, Median Blur (noise reduction), and CLAHE (Contrast Limited Adaptive Histogram Equalization) to amplify subtle calcified stone densities against renal parenchyma.
* **Training Specs:** Processed at 640×640 matrix resolution, utilizing Adam/SGD optimizers with extensive augmentations (random flips, mosaic, scaling).

### YOLO Architecture

![YOLO26 Architecture](<images/yolo_architecture.png>)

### Detection Results

![Model Evaluation and Results](<images/final evaluation.png>)
![Confusion Matrix](<images/confusion matrix.png>)
![YOLO26 Performance](<images/YOLO26 performance.png>)
![Predicted Image](<images/sucess case.png>)

---

## 🔬 Phase 2: Medical Image Segmentation (UNet++)

While bounding boxes provide localization, clinical burden measurement requires exact geometrical surface mapping. A UNet++ architecture is implemented to extract precise boundary definitions.

### Approach & Pipeline

* **Architecture:** UNet++ with a ResNet34 Encoder.
* **Weights:** Pre-trained on ImageNet to leverage robust low-level feature extraction capabilities.
* **Data Processing:** ROIs are resized to 512×512 and normalized. Augmentations include shifting, scaling, rotating (±2.5 degrees), and Gaussian noise injections to simulate clinical sensor variance.
* **Hyperparameters:** Trained for 100 epochs, batch size of 4, learning rate 0.0001, utilizing Dice Loss to counteract the severe foreground-background class imbalance inherent to small calculus segmentation.
* **Evaluation Metrics:** Evaluated on Intersection over Union (IoU / Jaccard Index), Dice Similarity Coefficient (DSC), Precision, and Recall.

### U-Net Architecture

![U-Net Architecture](<images/U-Net architecture.png>)

### Segmentation Visuals

![U-Net Metric Results](<images/Evaluation.png>)
![Metric Visuals](<images/metrix visuals.png>)
![Predicted Test Slices](<images/Test data result.png>)

---

## 💻 Technology & Framework Stack

* **Deep Learning:** PyTorch, Ultralytics YOLO, Segmentation Models PyTorch (SMP).
* **Computer Vision:** OpenCV (`cv2`), Albumentations.
* **Data Science:** Pandas, NumPy, Matplotlib, Scikit-learn.
* **Backend API & Deployment:** FastAPI, Uvicorn, Streamlit.
![Predicted normal(wrong)](<images/failure case.png>)

## ⚠️ Limitations

* **Localization Drop:** Performance falls at strict IoU thresholds (mAP@50-95 is 0.292).
* **False Negatives:** 24% of actual stones are missed and classified as background.
* **Scale Imbalance:** Stones are extremely small relative to the full CT scan.
* **Compute Cost:** Achieving higher recall requires double the training time and GFLOPs.


## 🚀 Future Work

* **Patch Inference:** Use SAHI (Slicing Aided Hyper Inference) to better detect tiny stones.
* **Loss Optimization:** Apply CIoU or NWD loss to improve bounding-box accuracy.
* **Deployment:** Export the efficient `yolo26s` model for real-time clinical use.
* **External Testing:** Validate the 97.08% Dice Score segmentation model on datasets from other hospitals.

🚀 How to Run the Application

1:  Clone the repository:
    git clone [https://github.com/HaiderCortex/kidney-stone-detection](https://github.com/HaiderCortex/kidney-stone-detection)
cd kidney-stone-detection

2:  Install dependencies:
    pip install -r requirements.txt

3:  Run the FastAPI Backend:
    uvicorn app:app --reload

4:  Run the Streamlit Frontend:
    streamlit run frontend.py