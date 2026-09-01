import cv2
import urllib.request
import os

# Ensure the XML cascade file exists locally
xml_file = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_file):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, xml_file)

face_cascade = cv2.CascadeClassifier(xml_file)

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        print("Failed to read camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()