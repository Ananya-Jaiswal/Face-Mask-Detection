# Real-Time Face Mask Detection

A real-time face mask detection system using deep learning and computer vision — capable of detecting faces and classifying whether they are wearing a mask or not from a webcam feed.


## Project Overview

Face mask detection became an important safety measure during the COVID-19 pandemic. This project implements a real-time mask detector that:
- Detects faces in live video from a webcam.
- Predicts whether each detected face is wearing a mask or not.
- Displays the class and confidence percentage on the video feed.

The system uses:
- **OpenCV** for webcam capture and face detection.
- **Convolutional Neural Network (CNN)** for mask/no-mask classification.
- **Haar Cascade** for fast face detection.
- **Keras/TensorFlow** for model loading and prediction.


## Features

- Real-time detection from webcam
- Face localization with bounding boxes
- Mask vs No Mask classification
- Confidence percentages displayed live
- Smooth and stable predictions


## Dataset

The dataset used: [Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

We trained the model on a dataset of approximately **~7,000+ labeled face images** covering:
- Faces with masks
- Faces without masks

The dataset was split into training and validation sets for proper model evaluation and robustness.


## Tech Stack

| Library / Tool | Purpose |
|----------------|---------|
| Python         | Programming language |
| OpenCV         | Face detection & webcam feed |
| TensorFlow / Keras | Deep learning model |
| NumPy          | Array processing |
| Haar Cascade   | Face detector model |


## Installation

1. Clone the repository:

```
git clone https://github.com/<your-username>/MaskDetection.git
cd MaskDetection
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run real-time detection:

```
python RealTimeMaskDetection.py
```

4. Real-time face mask detection is active and press 'X' or 'Q' to quit.


## Technical Implementation

- **Face Detection**: Uses OpenCV’s Haar Cascade classifier (haarcascade_frontalface_default.xml) for fast and lightweight frontal face detection in real-time video streams.
- **Image Preprocessing**: Detected face regions are cropped, resized to 100×100 pixels, normalized to a [0,1] range, and reshaped to match CNN input requirements.
- **Deep Learning Model**: Utilizes a custom Convolutional Neural Network (CNN) trained to classify faces into Mask and No Mask categories using softmax probabilities.
- **Real-Time Inference**: Integrates webcam feed via OpenCV and performs frame-by-frame prediction with bounding box visualization.
- **Confidence Visualization**: Displays predicted class labels along with confidence percentages directly on detected face regions.
- **Stability Handling**: Designed to provide smooth real-time predictions with consistent frame processing and graceful exit handling (keyboard and window close events).


## Future Improvements

- Replace Haar Cascade with a more robust face detector (SSD, MTCNN, or Mediapipe)
- Use transfer learning (MobileNet / EfficientNet) for higher accuracy
- Export model to TensorFlow Lite for mobile deployment
- Add alert sound or notification system for no-mask detection


## References

- OpenCV Haarcascade documentation
- Keras/TensorFlow model prediction tutorial
- [Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) from Kaggle
