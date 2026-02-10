import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []
label_map = {}
label_id = 0

FACE_SIZE = (200, 200)  # 👈 force same size

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if face is None:
            continue

        face = cv2.resize(face, FACE_SIZE)  # 🔥 FIX
        faces.append(face)
        labels.append(label_id)

    label_id += 1

faces = np.array(faces)
labels = np.array(labels)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)
model.save("face_model.yml")

print("Model trained successfully")
print("Label mapping:", label_map)
