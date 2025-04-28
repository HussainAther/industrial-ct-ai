# 🏭 Industrial-CT-AI
**AI & Computer Vision Toolkit for Automated Defect Detection in Industrial CT Scans**

---

## 🚀 Overview
**Industrial-CT-AI** is an open-source framework that combines **Computer Vision (CV)** techniques and **AI-powered models** to automatically detect structural defects in industrial components using CT scan imagery.

Whether you're analyzing engine parts, pipelines, or metal castings, this toolkit provides a streamlined pipeline for:
- 📈 **Pre-processing** CT images using OpenCV.
- 🤖 **Detecting defects** with YOLOv8 or custom CNN models.
- 🎨 **Visualizing results** through an interactive Streamlit dashboard.
- ⚙️ Generating synthetic CT phantoms for AI training and testing.

Originally inspired by advancements in medical CT reconstruction, this project adapts cutting-edge AI to the world of **automotive manufacturing**, **aerospace**, and **materials engineering**.

---

## ✨ Features
- **OpenCV Pre/Post-Processing:**  
   Enhance CT image quality, detect edges, highlight anomalies.

- **AI Defect Detection:**  
   Lightweight CNN and YOLOv8-based detection of cracks, voids, misalignments.

- **Interactive Dashboard:**  
   Upload CT images, run detection, visualize defects, export reports.

- **Synthetic Data Generator:**  
   Create labeled industrial CT phantoms with embedded defects for AI model development.

- **Modular & Extensible:**  
   Easy to plug in new models, CV techniques, or datasets.

---

## ⚡ Quickstart
```bash
git clone https://github.com/your-username/Industrial-CT-AI.git
cd Industrial-CT-AI
pip install -r requirements.txt

# Launch the interactive dashboard
streamlit run dashboard/app.py
```

---

## 🖼️ Example
<p align="center">
  <img src="docs/example_dashboard.png" width="700"/>
</p>

- **Left:** Reconstructed CT Image  
- **Right:** Detected defects (highlighted with bounding boxes & heatmaps)

---

## 📂 Project Structure
```plaintext
cv_utils/            # OpenCV-based image processing tools
defect_detection/    # AI models and inference wrappers
dashboard/           # Streamlit interactive web app
synthetic_phantoms/  # Generator for CT phantoms with defects
data/                # Sample CT images and annotations
```

---

## 🤖 AI Models
- **YOLOv8:** Fast object detection for cracks, voids.
- **CNN Classifier:** Lightweight defect detection option.
- Models tracked & managed via MLflow (compatible with HP AI Studio).

---

## 🚗 Industry Use Cases
- Automotive manufacturing: Engine block inspections
- Aerospace: Structural integrity of critical components
- Materials science: Quality control in metal casting & additive manufacturing

---

## 📜 License
MIT License

---

## 🤝 Contributing
We welcome contributions in:
- New CV algorithms
- AI model improvements
- Dashboard UI/UX enhancements
- Dataset expansions

---

## 📬 Contact
For questions or collaborations, reach out via [GitHub Issues](https://github.com/HussainAther/Industrial-CT-AI/issues) or email: `shussainather@gmail.com`

