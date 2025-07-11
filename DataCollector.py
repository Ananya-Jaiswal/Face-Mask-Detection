#For capturimg images from webcam and saving them in a folder (If not using the dataset provided in the repo)
import cv2

facedetect=cv2.CascadeClassifier(r"C:\Users\Hp\Desktop\Internship\MaskDetection\haarcascade_frontalface_default.xml")

video= cv2.VideoCapture(0)
 
count=0

while True:
    ret, frame = video.read()
    faces=facedetect.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), ( 0, 255, 0), 2)
    cv2.imshow("Webcam Feed", frame)
    k= cv2.waitKey(1)
    if k==ord("q"):
        break
video.release()
cv2.destroyAllWindows()

