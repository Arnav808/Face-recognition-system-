import cv2
import os

# Load face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.yml")

# Labels (folder names)
labels = os.listdir("dataset")

last_name = ""
same_count = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror fix

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = model.predict(face)

        if confidence < 80:
            predicted_name = labels[label].rstrip("0123456789")
        else:
            predicted_name = "Unknown"

        if predicted_name == last_name:
            same_count += 1
        else:
            same_count = 0
            last_name = predicted_name

        if same_count > 5:
            name = predicted_name
        else:
            name = "Unknown"

        # color based on known / unknown
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
