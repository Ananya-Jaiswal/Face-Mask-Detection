# Real-Time Face Mask Detection

A real-time face mask detection system using deep learning and computer vision — capable of detecting faces and classifying whether they are wearing a mask or not from a webcam feed.


## 🧠 Project Overview

Face mask detection became an important safety measure during the COVID-19 pandemic. This project implements a real-time mask detector that:
- Detects faces in live video from a webcam.
- Predicts whether each detected face is wearing a mask or not.
- Displays the class and confidence percentage on the video feed.

The system uses:
- **OpenCV** for webcam capture and face detection.
- **Convolutional Neural Network (CNN)** for mask/no-mask classification.
- **Haar Cascade** for fast face detection.
- **Keras/TensorFlow** for model loading and prediction.

---

## 🧩 Features

- 🔹 Real-time detection from webcam
- 🔹 Face localization with bounding boxes
- 🔹 Mask vs No Mask classification
- 🔹 Confidence percentages displayed live
- 🔹 Smooth and stable predictions

---

## 📁 Dataset

The dataset used: [Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

We trained the model on a dataset of approximately **~7,000+ labeled face images** covering:
- Faces with masks
- Faces without masks

The dataset was split into training and validation sets for proper model evaluation and robustness.

---

## 🛠️ Tech Stack

| Library / Tool | Purpose |
|----------------|---------|
| Python         | Programming language |
| OpenCV         | Face detection & webcam feed |
| TensorFlow / Keras | Deep learning model |
| NumPy          | Array processing |
| Haar Cascade   | Face detector model |

---

## 📌 Project Structure

