import cv2
import numpy as np
from keras.models import load_model

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(r"C:\Users\Hp\Desktop\Internship\MaskDetection\haarcascade_frontalface_default.xml")

# Load the pre-trained mask classification model
model = load_model(r"C:\Users\Hp\Desktop\Internship\MaskDetection\mask_classifier.h5")

# Start webcam capture
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)


    for (x, y, w, h) in faces:

        face_img= frame[y:y+h, x:x+w]

        resized = cv2.resize(face_img, (100, 100))

        normalized = resized / 255.0

        reshaped = np.reshape(normalized, (1, 100, 100, 3))

        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        prediction = model.predict(reshaped)
        label_index = np.argmax(prediction)

        if label_index == 0:
            label_text = "Mask"
            color = (0, 255, 0)

        else:
            label_text = "No Mask"
            color = (0, 0, 255)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        face=frame[y:y+h, x:x+w]
        resized_face=cv2.resize(face, (100, 100))
        normalized_face=resized_face/255.0
        reshaped_face=np.reshape(normalized_face, (1, 100, 100, 3))
        prediction=model.predict(reshaped_face)[0]

        mask_prob = prediction[0]
        no_mask_prob = prediction[1]

        if mask_prob > no_mask_prob:
            label = f"{mask_prob * 100:.1f}% Mask"
            color = (0, 255, 0)  
        else:
            label = f"{no_mask_prob * 100:.1f}% No Mask"
            color = (0, 0, 255)  

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
  
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()




