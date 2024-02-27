import face_recognition
import cv2
import numpy as np
import os
import glob

photos_folder = "data_unknown/"

trained_model = np.load("trained_model.npz")
known_face_encodings = trained_model['known_face_encodings']
known_face_names = trained_model['known_face_names']

face_locations = []
face_encodings = []
face_names = []

for image_path in glob.glob(os.path.join(photos_folder, '*.jpg')):
    frame = cv2.imread(image_path)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    print(f"Photo: {os.path.basename(image_path)}, Predicted Name(s): {', '.join(face_names)}")
